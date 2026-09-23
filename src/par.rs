//! Parallel execution for large scans.
//!
//! A scan is split into chunks of whole records, and workers claim chunks
//! dynamically, so faster cores take more of them. Callers only see
//! [`map_chunks`] and [`find_first_chunk`], which return results in chunk
//! order.
//!
//! The executor is a small persistent pool rather than a general work-stealing
//! scheduler. Idle workers spin for a moment before parking, so back-to-back
//! scans start in about 3 us instead of the 15-40 us it took to wake rayon's
//! workers. A job lives on the caller's stack and the caller works on it too,
//! so it never waits for a parked worker: it only waits for workers that have
//! joined the job. The pool runs one job at a time; concurrent or nested
//! callers run their chunks inline. A forked child gets a fresh pool.

use std::any::Any;
use std::mem::MaybeUninit;
use std::panic::{self, AssertUnwindSafe};
use std::ptr;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// A scan at the parallel threshold is split into this many chunks; larger
/// scans get proportionally more, up to `CHUNKS_PER_THREAD` per thread.
const CHUNKS_AT_THRESHOLD: usize = 8;
/// Smallest chunk worth claiming.
const MIN_CHUNK_BYTES: usize = 4 * 1024;
/// Chunks per worker thread, so a slow core cannot hold up the whole scan.
const CHUNKS_PER_THREAD: usize = 4;
/// Record counts per chunk are rounded to this so block scanners (16 records
/// per block) see no partial blocks except at the very end.
const RECORD_ALIGN: usize = 16;
/// How long an idle worker spins before parking. Back-to-back scans (the
/// common case in loops) then start without a wake-up.
const SPIN: Duration = Duration::from_micros(50);

/// Threads that work on a job, including the caller. `HEXHAMMING_NUM_THREADS`
/// overrides the default (available parallelism); `RAYON_NUM_THREADS` is
/// honored too, since it configured the previous executor.
#[inline]
pub(crate) fn threads() -> usize {
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(configured_threads)
}

fn configured_threads() -> usize {
    ["HEXHAMMING_NUM_THREADS", "RAYON_NUM_THREADS"]
        .iter()
        .find_map(|var| std::env::var(var).ok()?.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

// ---------------------------------------------------------------------------
// Pool
// ---------------------------------------------------------------------------

/// A job published to the workers. It lives on the caller's stack for the
/// whole of `Pool::run`.
/// Keeps a hot atomic on its own cache line (128 bytes covers Apple and
/// x86 prefetch pairs).
#[repr(align(128))]
struct Padded<T>(T);

struct Job {
    chunks: usize,
    next: Padded<AtomicUsize>,
    completed: Padded<AtomicUsize>,
    stop: AtomicBool,
    task: *const (dyn Fn(usize) + Sync),
    panic: Mutex<Option<Box<dyn Any + Send>>>,
}

impl Job {
    /// Claim and run chunks until none are left. After a panic the remaining
    /// chunks are claimed without running, so the job still completes.
    fn work(&self) {
        let mut done = 0;
        loop {
            let chunk = self.next.0.fetch_add(1, Ordering::Relaxed);
            if chunk >= self.chunks {
                if done > 0 {
                    self.completed.0.fetch_add(done, Ordering::Release);
                }
                return;
            }
            if !self.stop.load(Ordering::Relaxed) {
                // SAFETY: `task` outlives the job (see `Pool::run`).
                let task = unsafe { &*self.task };
                if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| task(chunk))) {
                    self.stop.store(true, Ordering::Relaxed);
                    let mut slot = self.panic.lock().unwrap_or_else(|e| e.into_inner());
                    slot.get_or_insert(payload);
                }
            }
            done += 1;
        }
    }
}

struct Pool {
    workers: AtomicUsize,
    pid: u32,
    busy: AtomicBool,
    generation: AtomicU64,
    job: AtomicPtr<Job>,
    /// Workers that may be reading `job`.
    users: AtomicUsize,
    parked: AtomicUsize,
    wake: Mutex<u64>,
    wake_cv: Condvar,
}

impl Pool {
    fn start(threads: usize) -> &'static Pool {
        let pool: &'static Pool = Box::leak(Box::new(Pool {
            workers: AtomicUsize::new(0),
            pid: std::process::id(),
            busy: AtomicBool::new(false),
            generation: AtomicU64::new(0),
            job: AtomicPtr::new(ptr::null_mut()),
            users: AtomicUsize::new(0),
            parked: AtomicUsize::new(0),
            wake: Mutex::new(0),
            wake_cv: Condvar::new(),
        }));
        for id in 1..threads {
            let spawned = std::thread::Builder::new()
                .name(format!("hexhamming-{id}"))
                .spawn(move || pool.worker());
            if spawned.is_err() {
                break;
            }
            pool.workers.fetch_add(1, Ordering::Relaxed);
        }
        pool
    }

    fn worker(&'static self) {
        let mut seen = self.generation.load(Ordering::Acquire);
        loop {
            seen = self.wait_for_job(seen);
            self.users.fetch_add(1, Ordering::SeqCst);
            let job = self.job.load(Ordering::SeqCst);
            if !job.is_null() {
                // SAFETY: the caller keeps the job alive until it has cleared
                // `job` and seen `users` drop to zero.
                unsafe { (*job).work() };
            }
            self.users.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// Spin, then park, until the generation moves past `seen`.
    fn wait_for_job(&self, seen: u64) -> u64 {
        let start = Instant::now();
        let mut spins = 0u32;
        loop {
            let generation = self.generation.load(Ordering::Acquire);
            if generation != seen {
                return generation;
            }
            spins = spins.wrapping_add(1);
            if spins % 64 == 0 && start.elapsed() > SPIN {
                let mut published = self.wake.lock().unwrap_or_else(|e| e.into_inner());
                self.parked.fetch_add(1, Ordering::SeqCst);
                while *published == seen {
                    published = self
                        .wake_cv
                        .wait(published)
                        .unwrap_or_else(|e| e.into_inner());
                }
                self.parked.fetch_sub(1, Ordering::SeqCst);
                return *published;
            }
            std::hint::spin_loop();
        }
    }

    /// Run `task(chunk)` for every chunk on the caller and the workers. Returns
    /// `false` without running anything if the pool is busy.
    fn run(&self, chunks: usize, task: &(dyn Fn(usize) + Sync)) -> bool {
        if self
            .busy
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }
        // SAFETY: only the lifetime is erased; `task` outlives this call and
        // the job is unpublished (and unused) before it returns.
        let task: *const (dyn Fn(usize) + Sync) = unsafe { std::mem::transmute(task) };
        let job = Job {
            chunks,
            next: Padded(AtomicUsize::new(0)),
            completed: Padded(AtomicUsize::new(0)),
            stop: AtomicBool::new(false),
            task,
            panic: Mutex::new(None),
        };
        self.job
            .store(&job as *const Job as *mut Job, Ordering::SeqCst);
        let generation = self.generation.fetch_add(1, Ordering::Release) + 1;
        {
            // Workers register as parked under this lock, so checking here
            // cannot miss one that is about to wait.
            let mut published = self.wake.lock().unwrap_or_else(|e| e.into_inner());
            *published = generation;
            if self.parked.load(Ordering::SeqCst) > 0 {
                self.wake_cv.notify_all();
            }
        }

        job.work();
        let mut spins = 0u32;
        while job.completed.0.load(Ordering::Acquire) < chunks {
            spins = spins.wrapping_add(1);
            if spins % 1024 == 0 {
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        self.job.store(ptr::null_mut(), Ordering::SeqCst);
        while self.users.load(Ordering::SeqCst) != 0 {
            std::hint::spin_loop();
        }
        self.busy.store(false, Ordering::Release);

        if let Some(payload) = job.panic.into_inner().unwrap_or_else(|e| e.into_inner()) {
            panic::resume_unwind(payload);
        }
        true
    }
}

static POOL: Mutex<Option<&'static Pool>> = Mutex::new(None);

/// The pool for this process, started on first use, or `None` when only one
/// thread is configured. A forked child, whose copy has no worker threads,
/// starts its own.
fn pool() -> Option<&'static Pool> {
    let mut slot = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let pid = std::process::id();
    match *slot {
        Some(pool) if pool.pid == pid => {}
        _ => {
            let threads = configured_threads();
            if threads < 2 {
                return None;
            }
            *slot = Some(Pool::start(threads));
        }
    }
    slot.filter(|pool| pool.workers.load(Ordering::Relaxed) > 0)
}

/// Run `task` for every chunk, in parallel when the pool is free.
fn for_each_chunk(chunks: usize, task: &(dyn Fn(usize) + Sync)) {
    if chunks > 1 {
        if let Some(pool) = pool() {
            if pool.run(chunks, task) {
                return;
            }
        }
    }
    for chunk in 0..chunks {
        task(chunk);
    }
}

// ---------------------------------------------------------------------------
// Plans and chunked execution
// ---------------------------------------------------------------------------

/// How to split `records` records across chunks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Plan {
    pub(crate) records_per_chunk: usize,
    pub(crate) chunks: usize,
}

impl Plan {
    /// Record range of `chunk`, clamped to `records`.
    #[inline]
    pub(crate) fn range(self, chunk: usize, records: usize) -> (usize, usize) {
        let start = chunk * self.records_per_chunk;
        (start, (start + self.records_per_chunk).min(records))
    }
}

/// Chunk plan for scanning `records` records of `width` bytes, or `None` when
/// the scan (under `min_bytes`) is too small to be worth splitting.
pub(crate) fn plan(records: usize, width: usize, min_bytes: usize) -> Option<Plan> {
    if records.saturating_mul(width) < min_bytes {
        return None;
    }
    split(records, width, min_bytes)
}

/// Chunk plan sized for a parallel threshold of `min_bytes` (a scan of that
/// size gets `CHUNKS_AT_THRESHOLD` chunks), without requiring this scan to
/// reach it. `None` if there is too little to split or only one thread.
pub(crate) fn split(records: usize, width: usize, min_bytes: usize) -> Option<Plan> {
    let bytes = records.saturating_mul(width);
    if records < 2 * RECORD_ALIGN {
        return None;
    }
    let threads = threads();
    if threads < 2 {
        return None;
    }
    let chunk_bytes = (min_bytes / CHUNKS_AT_THRESHOLD).max(MIN_CHUNK_BYTES);
    let chunks = (bytes / chunk_bytes).clamp(2, threads * CHUNKS_PER_THREAD);
    let records_per_chunk = records.div_ceil(chunks).next_multiple_of(RECORD_ALIGN);
    Some(Plan {
        records_per_chunk,
        chunks: records.div_ceil(records_per_chunk),
    })
}

/// Chunk plan for running `items` independent jobs of `bytes_per_item` bytes
/// each, or `None` when the total is under `min_bytes`.
pub(crate) fn plan_items(items: usize, bytes_per_item: usize, min_bytes: usize) -> Option<Plan> {
    let bytes = items.saturating_mul(bytes_per_item);
    if items < 2 || bytes < min_bytes {
        return None;
    }
    let threads = threads();
    if threads < 2 {
        return None;
    }
    let chunk_bytes = (min_bytes / CHUNKS_AT_THRESHOLD).max(MIN_CHUNK_BYTES);
    let chunks = (bytes / chunk_bytes).clamp(2, items.min(threads * CHUNKS_PER_THREAD));
    let per_chunk = items.div_ceil(chunks);
    Some(Plan {
        records_per_chunk: per_chunk,
        chunks: items.div_ceil(per_chunk),
    })
}

/// Result slots written by exactly one chunk each.
struct Slots<R>(Vec<MaybeUninit<R>>);

// SAFETY: every slot is written by exactly one chunk and read only after the
// job has completed.
unsafe impl<R: Send> Sync for Slots<R> {}

impl<R> Slots<R> {
    fn new(len: usize) -> Self {
        let mut slots = Vec::with_capacity(len);
        slots.resize_with(len, MaybeUninit::uninit);
        Self(slots)
    }

    /// SAFETY: each index must be written exactly once, and not concurrently
    /// with reads.
    unsafe fn write(&self, index: usize, value: R) {
        (self.0.as_ptr().add(index) as *mut MaybeUninit<R>)
            .as_mut()
            .unwrap_unchecked()
            .write(value);
    }

    /// SAFETY: every slot must have been written.
    unsafe fn into_vec(self) -> Vec<R> {
        self.0.into_iter().map(|slot| slot.assume_init()).collect()
    }
}

/// `f(chunk)` for every chunk, in chunk order.
pub(crate) fn map_chunks<R, F>(chunks: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync,
{
    let slots = Slots::new(chunks);
    // SAFETY: chunk indices are claimed exactly once each.
    for_each_chunk(chunks, &|chunk| unsafe { slots.write(chunk, f(chunk)) });
    // SAFETY: `for_each_chunk` returns only after every chunk ran (a panic
    // unwinds past this point instead).
    unsafe { slots.into_vec() }
}

/// The result of the lowest-numbered chunk for which `f` returns `Some`.
/// Chunks after one that has returned `Some` may be skipped.
pub(crate) fn find_first_chunk<R, F>(chunks: usize, f: F) -> Option<R>
where
    R: Send,
    F: Fn(usize) -> Option<R> + Sync,
{
    let found = AtomicUsize::new(usize::MAX);
    map_chunks(chunks, |chunk| {
        if found.load(Ordering::Relaxed) < chunk {
            return None;
        }
        let result = f(chunk);
        if result.is_some() {
            found.fetch_min(chunk, Ordering::Relaxed);
        }
        result
    })
    .into_iter()
    .flatten()
    .next()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plans_cover_every_record_exactly_once() {
        for records in [32usize, 33, 1000, 4096, 100_003, 1 << 20] {
            for width in [1usize, 8, 16, 20, 64] {
                let Some(plan) = plan(records, width, 0) else {
                    continue;
                };
                let mut next = 0;
                for chunk in 0..plan.chunks {
                    let (start, end) = plan.range(chunk, records);
                    assert_eq!(start, next);
                    assert!(end > start, "empty chunk {chunk} of {plan:?}");
                    next = end;
                }
                assert_eq!(next, records);
                assert_eq!(plan.records_per_chunk % RECORD_ALIGN, 0);
            }
        }
        assert_eq!(plan(1 << 20, 16, usize::MAX), None);
    }

    #[test]
    fn item_plans_cover_every_item_exactly_once() {
        if threads() < 2 {
            return;
        }
        for items in [2usize, 3, 17, 64, 1000] {
            let plan = plan_items(items, 1 << 20, 0).unwrap();
            let covered: usize = (0..plan.chunks)
                .map(|c| {
                    let (s, e) = plan.range(c, items);
                    e - s
                })
                .sum();
            assert_eq!(covered, items);
        }
        assert_eq!(plan_items(1, 1 << 30, 0), None);
    }

    #[test]
    fn panics_propagate_and_the_pool_stays_usable() {
        let result = std::panic::catch_unwind(|| {
            map_chunks(64, |chunk| {
                if chunk == 37 {
                    panic!("chunk 37");
                }
                chunk
            })
        });
        let payload = result.expect_err("panic must reach the caller");
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"chunk 37"));
        assert_eq!(map_chunks(64, |c| c + 1), (1..=64).collect::<Vec<_>>());
    }

    #[test]
    fn concurrent_and_nested_callers_get_correct_results() {
        let (threads, rounds) = if cfg!(miri) { (3, 3) } else { (8, 50) };
        std::thread::scope(|scope| {
            for t in 0..threads {
                scope.spawn(move || {
                    for round in 0..rounds {
                        let chunks = 1 + (t * 7 + round) % 97;
                        let got = map_chunks(chunks, |c| {
                            // Nested jobs run inline instead of deadlocking.
                            let inner: usize = map_chunks(3, |i| i + c).into_iter().sum();
                            (c, inner)
                        });
                        let want: Vec<_> = (0..chunks).map(|c| (c, 3 * c + 3)).collect();
                        assert_eq!(got, want);
                    }
                });
            }
        });
    }

    #[test]
    fn back_to_back_jobs_complete() {
        // Exercises the spin/park hand-off: some jobs start while workers
        // spin, others after they have parked.
        let rounds = if cfg!(miri) { 12 } else { 2000 };
        for round in 0..rounds {
            let chunks = 2 + round % 61;
            let sum: usize = map_chunks(chunks, |c| c).into_iter().sum();
            assert_eq!(sum, chunks * (chunks - 1) / 2);
            if round % 500 == 499 || (cfg!(miri) && round % 4 == 3) {
                std::thread::sleep(SPIN * 3);
            }
        }
    }

    #[test]
    fn chunk_helpers_keep_order() {
        assert_eq!(
            map_chunks(100, |c| c * 2),
            (0..100).map(|c| c * 2).collect::<Vec<_>>()
        );
        assert_eq!(
            find_first_chunk(100, |c| (c % 7 == 3).then_some(c)),
            Some(3)
        );
        assert_eq!(find_first_chunk(100, |_| None::<usize>), None);
    }
}
