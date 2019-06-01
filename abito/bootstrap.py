import numpy as np
import multiprocessing


def generate_sample(observations: np.ndarray, n_iters: int):
    ids = np.random.choice(
        observations.shape[0],
        (n_iters, observations.shape[0]),
        replace=True,
    )
    return observations[ids]


def get_bootstrap_batch(
        observations: np.ndarray,
        n_iters: int,
        iters_batch_size: int,
        seed_low: int,
        seed_high: int,
):
    results = []
    for rng in range(0, n_iters, iters_batch_size):
        seed = np.random.randint(seed_low, seed_high)
        np.random.seed(seed)
        iters_todo = min(iters_batch_size, n_iters - rng)
        values_sim = generate_sample(observations, iters_todo)
        sum_values = np.sum(values_sim, axis=1)
        r = np.divide(sum_values[..., 0], sum_values[..., 1])
        results.append(r)
    return np.concatenate(results, axis=0)


def get_bootstrap_dist(
        num: np.ndarray,
        den: np.ndarray,
        n_iters: int,
        iters_batch_size: int,
        n_threads: int,
):
    n_iters = int(n_iters)
    iters_batch_size = int(iters_batch_size)
    n_threads = int(n_threads)

    n_threads = multiprocessing.cpu_count() if n_threads == -1 else n_threads

    observations = np.vstack((num, den)).T
    if n_threads <= 1:
        results = get_bootstrap_batch(observations, n_iters, iters_batch_size, 0, 2 ** 32)
    else:
        with multiprocessing.Pool(n_threads) as pool:
            iters_per_thread = int(np.ceil(n_iters * 1.0 / n_threads))
            results = []
            seed_batch_size = np.ceil(2 ** 32 / n_threads)
            for seed_low in np.random.permutation(n_threads) * seed_batch_size:
                r = pool.apply_async(
                    get_bootstrap_batch,
                    (observations, iters_per_thread, iters_batch_size, seed_low, seed_low + seed_batch_size))
                results.append(r)

            results = np.concatenate([res.get() for res in results], axis=0)

    return results
