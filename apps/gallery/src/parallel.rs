use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    thread,
};

pub fn available_workers() -> usize {
    thread::available_parallelism().map_or(1, usize::from)
}

/// Apply `operation` concurrently and report each result as soon as it is
/// ready. Input stays borrowed by the calling thread, which avoids cloning a
/// growing in-memory capture merely to distribute CPU work.
pub fn for_each<T, R>(
    items: &[T],
    worker_limit: usize,
    operation: impl Fn(&T) -> R + Sync,
    mut on_result: impl FnMut(usize, R),
) where
    T: Sync,
    R: Send,
{
    let worker_count = worker_limit.max(1).min(items.len());
    if worker_count <= 1 {
        for (index, item) in items.iter().enumerate() {
            on_result(index, operation(item));
        }
        return;
    }

    let next = AtomicUsize::new(0);
    thread::scope(|scope| {
        let (result_sender, result_receiver) = mpsc::channel();
        for _ in 0..worker_count {
            let result_sender = result_sender.clone();
            let operation = &operation;
            let next = &next;
            scope.spawn(move || {
                loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    let Some(item) = items.get(index) else {
                        break;
                    };
                    if result_sender.send((index, operation(item))).is_err() {
                        break;
                    }
                }
            });
        }
        drop(result_sender);
        for (index, result) in result_receiver {
            on_result(index, result);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{collections::HashSet, sync::Barrier};

    #[test]
    fn parallel_for_each_reports_every_result_with_its_index() {
        let input = [1, 2, 3, 4, 5, 6];
        let mut output = vec![0; input.len()];
        for_each(
            &input,
            4,
            |value| value * value,
            |index, result| output[index] = result,
        );
        assert_eq!(output, [1, 4, 9, 16, 25, 36]);
    }

    #[test]
    fn parallel_for_each_handles_empty_input() {
        for_each::<u8, u8>(
            &[],
            4,
            |value| *value,
            |_, _| panic!("empty input produced a result"),
        );
    }

    #[test]
    fn parallel_for_each_uses_the_requested_workers() {
        let barrier = Barrier::new(4);
        let thread_ids = std::sync::Mutex::new(HashSet::new());
        for_each(
            &[0, 1, 2, 3],
            4,
            |value| {
                thread_ids
                    .lock()
                    .expect("thread id set poisoned")
                    .insert(thread::current().id());
                barrier.wait();
                *value
            },
            |_, _| {},
        );
        assert_eq!(thread_ids.lock().expect("thread id set poisoned").len(), 4);
    }
}
