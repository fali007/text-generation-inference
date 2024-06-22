use crate::pb::fmaas::RunningParamsInfoResponse;
use std::collections::BinaryHeap;
use std::sync::Mutex;
use std::{
    collections::BTreeSet,
    mem::take,
    time::Duration,
    sync::Arc,
};

use axum::http::request;
use nohash_hasher::IntMap;
use text_generation_client::{
    Batch, ClientError, LengthPenalty, NextTokenChooserParameters, Request, RequestedDetails, Token,
};
use tokio::{
    sync::{
        mpsc::{
            error::TryRecvError::{Disconnected, Empty},
            Receiver, UnboundedSender,
        },
        oneshot::Sender,
    },
    time::Instant,
};
use tracing::info;

use crate::{
    batch_types::BatchType, batcher::InferResponse, decoder::IncrementalDecoderWrapper,
    GenerateParameters, GenerateRequest,
};
use crate::metrics::{increment_labeled_counter, observe_histogram, set_gauge};

// Requests that fit into the next batch can overtake others
// that don't as long as they arrive within this amount of time after
const CUTOFF_DURATION: Duration = Duration::from_secs(1);

#[derive(Debug)]
pub struct PriorityQueue {
    heap: BinaryHeap<Entry>,
}
impl PriorityQueue {
	pub fn new(capacity: usize) -> Self {
		let heap = BinaryHeap::with_capacity(capacity);
		Self{ heap }
	}

	pub fn push(&mut self, value: Entry) {
		self.heap.push(value)
	}

	pub fn pop(&mut self) -> Option<Entry> {
		return self.heap.pop()
	}

    pub fn drain(&mut self, size: usize) -> Vec<Entry> {
        let mut res: Vec<Entry> = Vec::new();
        for _e in 0..size {
            res.push(self.pop().unwrap())
        }
        res
    }

    pub fn peek(&mut self) -> Option<&Entry> {
        return self.heap.peek()
    }
}
impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
       Some(self.cmp(other))
    }
}
impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}
impl Eq for Entry {}
impl Ord for Entry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // This is for min heap >> highest priority user is priority 1, lowest is <intmax>.
        // To convert to maxheap interchange left and right variables.
        let left = other.priority;
        let right = self.priority;
        left.cmp(&right)
    }
}

/// Queue entry / in-progress request state
#[derive(Debug)]
pub(crate) struct Entry {
    /// Request
    pub request: GenerateRequest,
    /// Response senders to communicate between the Batcher and the batching_task
    /// Exactly one of these will be non-None
    pub response_tx: Option<Sender<Result<InferResponse, ClientError>>>,
    pub stream_tx: Option<UnboundedSender<Result<InferResponse, ClientError>>>,
    /// Number of tokens in the input
    pub input_length: usize,
    /// Number of virtual tokens in the prefix, if one is specified
    pub prefix_length: usize,
    /// Instant when this entry was queued
    pub queue_time: Instant,
    /// Instant when this entry was added to a batch (queue end time)
    pub batch_time: Option<Instant>,
    /// Generated token ids, populated only in non-streaming case
    pub token_ids: Vec<u32>,
    /// Generated tokens
    pub tokens: Vec<Token>,
    /// Input tokens
    pub input_tokens: Vec<Token>,
    /// Accumulates output, used only when stop sequences are provided
    pub output: Option<IncrementalDecoderWrapper>,
    /// Generated token count
    pub generated_tokens: u32,
    pub priority: u32,
}

impl Entry {
    pub(crate) fn new(
        request: GenerateRequest,
        input_length: usize,
        prefix_length: usize,
        response_tx: Option<Sender<Result<InferResponse, ClientError>>>,
        stream_tx: Option<UnboundedSender<Result<InferResponse, ClientError>>>,
        priority: u32,
    ) -> Self {
        Self {
            request,
            response_tx,
            stream_tx,
            input_length,
            prefix_length,
            input_tokens: vec![],
            queue_time: Instant::now(),
            batch_time: None,
            token_ids: vec![],
            tokens: vec![],
            output: None,
            generated_tokens: 0,
            priority,
        }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        if self.response_tx.is_some() {
            self.response_tx.as_ref().unwrap().is_closed()
        } else {
            self.stream_tx.as_ref().unwrap().is_closed()
        }
    }

    pub(crate) fn deadline_exceeded(&self) -> bool {
        matches![self.request.parameters.deadline, Some(d) if d < Instant::now()]
    }

    // Convenience method for sending a terminating response
    pub(crate) fn send_final(
        &mut self,
        result: Result<InferResponse, ClientError>,
    ) -> Result<(), Result<InferResponse, ClientError>> {
        if self.response_tx.is_some() {
            let rtx = take(&mut self.response_tx);
            rtx.unwrap().send(result)
        } else {
            self.stream_tx
                .as_mut()
                .unwrap()
                .send(result)
                .map_err(|s| s.0)
        }
    }
}

#[derive(Debug)]
pub(crate) struct BatchingConfig {
    /// Upper bound on number of requests in a batch
    pub(crate) size_limit: usize,
    /// Maximum batch "weight" at any point of time (takes sequence lengths into account)
    pub(crate) weight_limit: usize,
    /// Maximum percentage of pad tokens in prefill batches. In range [0, 1]
    pub(crate) prefill_padding_limit: f32,
}

/// Request Queue
#[derive(Debug)]
pub(crate) struct Queue<B: BatchType> {
    /// Batching config
    config: BatchingConfig,
    /// Batch type
    batch_type: B,

    receiver: Receiver<Vec<Entry>>,
    // Staging buffer, filled until max_size is reached
    buffer: PriorityQueue,
    /// Id of the next entry
    next_id: u64,
    /// Id of the next batch
    next_batch_id: u64,

    // Keep track what was logged in the last call to try_next_batch
    // so as to avoid many repeating entries in the log
    last_logged: Option<(usize, usize)>,

    /// Just a constant empty map to reuse
    empty_map: IntMap<u64, Entry>,
    running_params: Arc<Mutex<RunningParamsInfoResponse>>
}

impl<B: BatchType> Queue<B> {
    pub(crate) fn new(
        config: BatchingConfig,
        _batch_type: B,
        receiver: Receiver<Vec<Entry>>,
        running_params: Arc<Mutex<RunningParamsInfoResponse>>,
    ) -> Self {
        Self {
            config,
            receiver,
            buffer: PriorityQueue::new(1000),
            next_id: 0,
            next_batch_id: 1,
            batch_type: _batch_type,
            last_logged: None,
            empty_map: IntMap::default(),
            running_params: running_params,
        }
    }

    /// Get the next batch, blocking until available
    /// Corresponding entries are added to the entries map
    /// Returns None only if the queue has been closed
    pub(crate) async fn next_batch(&mut self, entries: &mut IntMap<u64, Entry>) -> Option<Batch> {
        loop {
            if self.buffer.heap.is_empty() {
                // Await on the queue while the buffer is empty
                match self.receiver.recv().await {
                    Some(ents) => self.add_to_buffer(ents),
                    // Queue closed, we must be shutting down
                    None => return None,
                }
                loop {
                    match self.receiver.try_recv() {
                        Ok(ents) => self.add_to_buffer(ents),
                        Err(Empty) => break,
                        Err(Disconnected) => return None,
                    }
                }
            }
            // We have at least one entry in the buffer
            if let Some(batch) = self.try_next_batch(entries, 1) {
                return Some(batch);
            }
        }
    }

    /// Returns a future that can be awaited to consume requests from the queue's
    /// shared channel into it's internal buffer. The future never completes.
    pub(crate) async fn service_queue(&mut self) {
        // First prune existing cancelled or expired requests
        let mut pruned = false;
        self.buffer.heap.retain(|entry| match entry {
            entry if entry.is_cancelled() => {
                increment_labeled_counter("tgi_request_failure", &[("err", "cancelled")], 1);
                pruned = true;
                false
            }
            entry if entry.deadline_exceeded() => {
                // Send timeout response
                increment_labeled_counter("tgi_request_failure", &[("err", "timeout")], 1);
                // entry.batch_time = Some(Instant::now());
                // entry.send_final(Ok(InferResponse::early_timeout(entry))).unwrap_or_default();
                pruned = true;
                false
            }
            _ => true,
        });

        if pruned {
            set_gauge("tgi_queue_size", self.buffer.heap.len() as f64);
        }

        while let Some(ents) = self.receiver.recv().await {
            self.add_to_buffer(ents);
        }
    }

    fn add_to_buffer(&mut self, new_entries: Vec<Entry>) {
        for req in new_entries {
            self.buffer.push(req);
        }
        set_gauge("tgi_queue_size", self.buffer.heap.len() as f64);
    }

    #[allow(dead_code)]
    fn get_queue_length(&mut self) -> usize{
        return self.buffer.heap.len()
    }

    /// Get the next batch without blocking.
    /// Corresponding entries are added to the entries map
    pub(crate) fn try_next_batch(
        &mut self,
        entries: &mut IntMap<u64, Entry>,
        min_size: usize,
    ) -> Option<Batch> {
        let buffer_size = self.buffer.heap.len();
        if buffer_size < min_size {
            // Not enough requests waiting to reach min_size
            self.last_logged = None;
            return None;
        }

        let mut total_count = entries.len();
        if total_count + min_size > self.config.size_limit {
            // Not enough space to fit min_size within max batch size
            self.last_logged = None;
            return None;
        }

        // Indices into buffer of entries chosen to add to next batch
        let mut btree = None;

        let now = Instant::now();
        let batch_stats = <B>::compute_stats(entries);
        let prefill_stats = <B>::compute_stats(&self.empty_map);

        // Compute the effective prefill weight limit, taking into account space already consumed
        // by the in-progress batch
        let effective_prefill_weight_limit = match self.config.weight_limit {
            prefill_limit if prefill_limit == 0 || total_count == 0 => prefill_limit,
            prefill_limit => {
                let current_batch_weight = self
                    .batch_type
                    .batch_initial_weight(&batch_stats, total_count);
                let pct_space_free =
                    1.0 - (current_batch_weight as f64 / self.config.weight_limit as f64) * 1.2;
                let limit = (pct_space_free * prefill_limit as f64) as usize;
                if limit == 0 {
                    return None;
                }
                limit
            }
        };
        let max_prefill_padding = self.config.prefill_padding_limit;

        let tree = btree.get_or_insert_with(|| {
            let mut t = Box::<BTreeSet<(usize, usize, usize)>>::default();
            for (_, e) in entries.iter() {
                let generated_count = e.generated_tokens as usize;
                t.insert((e.request.parameters.max_new_tokens as usize - generated_count,e.input_length + e.prefix_length + generated_count,t.len()));
            }
            t
        });

        // enumerate over each req to see if it fits into the batch
        // If possible add request to the batch else stop.

        let mut choosen_req: Vec<Entry> = Vec::new();

        for _index in 0..buffer_size {
            let entry = self.buffer.peek().unwrap();
            let input_len = entry.input_length + entry.prefix_length;
            let output_len = entry.request.parameters.max_new_tokens as usize;
            let next_stats = <B>::update_stats(&batch_stats, input_len * 1.2 as usize, output_len * 1.2 as usize);

            tree.insert((output_len, input_len, tree.len()));

            if self.batch_type.batch_max_weight(&next_stats, total_count + 1) > self.config.weight_limit {
                if choosen_req.len() == 0 {
                    return None
                } else {
                    break
                }
            } else if !tree.is_empty() {
                tree.insert((output_len, input_len, tree.len()));
            }
            if effective_prefill_weight_limit > 0 || max_prefill_padding < 1.0 {
                let next_prefill_stats = <B>::update_stats(&prefill_stats, input_len, 0);
                let batch_size = choosen_req.len() + 1;
                if effective_prefill_weight_limit > 0 {
                    let prefill_weight = self.batch_type.prefill_weight(&next_prefill_stats, batch_size);
                    if prefill_weight > effective_prefill_weight_limit {
                        break
                    }
                }
            }
            total_count += 1;
            if total_count >= self.config.size_limit {
                break;
            }
            choosen_req.push(self.buffer.pop().unwrap());
        }

        let chosen_count = choosen_req.len();
        if chosen_count == 0 {
            // Don't repeatedly log when no requests were chosen if the current/waiting
            // request counts haven't changed
            let current_counts = Some((buffer_size, total_count));
            if self.last_logged != current_counts {
                self.last_logged = current_counts;
                info!("Chose 0 out of {buffer_size} requests from buffer, total now {total_count}");
            }
            return None;
        }

        self.last_logged = None;
        info!(
            "Chose {chosen_count} out of {buffer_size} requests from buffer, \
                total now {total_count}"
        );

        let some_now = Some(now);
        let mut requests : Vec<Request> = Vec::new();
        for mut entry in choosen_req {
            let id = self.next_id;
            self.next_id += 1;
            let request = Request {
                id,
                prefix_id: entry.request.prefix_id.clone().unwrap_or_default(),
                inputs: entry.request.inputs.clone(),
                input_length: entry.input_length as u32,
                max_output_length: entry.request.parameters.max_new_tokens,
                truncate: entry.request.parameters.truncate_input_tokens > 0,
                parameters: Some((&entry.request.parameters).into()),
                stream_response: entry.stream_tx.is_some(),
                details: (&entry.request.parameters).into(),
            };
            entry.batch_time = some_now;
            observe_histogram(
                "tgi_request_queue_duration",
                (now - entry.queue_time).as_secs_f64()
            );
            entries.insert(id, entry);
            requests.push(request);
        }

        let batch_tokens = <B>::count_tokens(
            requests.iter().map(|r| r.input_length as usize),
            chosen_count,
        );
        set_gauge("tgi_queue_size", self.buffer.heap.len() as f64);
        let batch = Batch {
            id: self.next_batch_id,
            requests,
            total_tokens: batch_tokens as u32,
        };
        // Increment batch id
        self.next_batch_id += 1;
        Some(batch)
    }
}

impl From<&GenerateParameters> for NextTokenChooserParameters {
    fn from(parameters: &GenerateParameters) -> Self {
        Self {
            temperature: parameters.temperature,
            top_k: parameters.top_k as u32,
            top_p: parameters.top_p,
            typical_p: parameters.typical_p,
            min_new_tokens: parameters.min_new_tokens,
            seed: parameters.seed,
            repetition_penalty: match parameters.repetition_penalty {
                x if x == 1.0 || x == 0.0 => None,
                theta => Some(theta),
            },
            length_penalty: parameters
                .length_penalty
                .map(|(start_index, decay_factor)| LengthPenalty {
                    start_index,
                    decay_factor,
                }),
        }
    }
}

impl From<&GenerateParameters> for Option<RequestedDetails> {
    fn from(parameters: &GenerateParameters) -> Self {
        Some(RequestedDetails {
            input_toks: parameters.include_input_tokens,
            logprobs: parameters.include_logprobs,
            ranks: parameters.include_ranks,
            top_n_toks: parameters.include_top_n,
        })
    }
}
