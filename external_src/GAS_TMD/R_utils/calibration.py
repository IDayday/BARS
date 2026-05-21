import json
import os

import numpy as np


def quantile_key(prefix, q):
    return f"{prefix}_q{int(round(float(q) * 100)):02d}"


def calibrate_tmd_scales(
    dataset,
    provider,
    terminals,
    temporal_horizon_steps,
    sample_size=50000,
    quantiles=(0.5, 0.75, 0.9),
    seed=0,
):
    """
    Calibrate representation and TMD distance scales from same-trajectory pairs.

    Samples (s_t, s_{t+H}) pairs that do not cross trajectory terminals and
    returns quantile-based thresholds.  The temporal horizon is only used to
    choose pairs; it is not reused as a distance cutoff.
    """
    rng = np.random.default_rng(seed)
    observations = np.asarray(dataset['observations'])
    terminals = np.asarray(terminals)
    terminal_locs = np.flatnonzero(terminals > 0)
    if len(terminal_locs) == 0 or terminal_locs[-1] != len(observations) - 1:
        raise ValueError("Terminals must mark trajectory ends and include the final dataset index.")

    all_idxs = np.arange(len(observations) - int(temporal_horizon_steps), dtype=np.int64)
    final_state_idxs = terminal_locs[np.searchsorted(terminal_locs, all_idxs)]
    valid = all_idxs + int(temporal_horizon_steps) <= final_state_idxs
    valid_idxs = all_idxs[valid]
    if len(valid_idxs) == 0:
        raise ValueError(f"No valid same-trajectory pairs for H={temporal_horizon_steps}.")

    n = min(int(sample_size), len(valid_idxs))
    replace = n > len(valid_idxs)
    src_idxs = rng.choice(valid_idxs, size=n, replace=replace)
    dst_idxs = src_idxs + int(temporal_horizon_steps)

    src_embeds = provider.encode(observations[src_idxs])
    dst_embeds = provider.encode(observations[dst_idxs])
    repr_l2 = np.linalg.norm(dst_embeds - src_embeds, axis=-1)

    tmd_dist = np.zeros(n, dtype=np.float32)
    batch_size = provider.batch_size
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        tmd_dist[start:end] = np.asarray(
            provider.agent.get_tmd_distance_from_embeddings(src_embeds[start:end], dst_embeds[start:end]),
            dtype=np.float32,
        )

    result = {"temporal_horizon_steps": int(temporal_horizon_steps), "num_pairs": int(n)}
    for q in quantiles:
        result[quantile_key("repr_l2", q)] = float(np.quantile(repr_l2, q))
        result[quantile_key("tmd_dist", q)] = float(np.quantile(tmd_dist, q))

    result["repr_cluster_threshold"] = float(result.get("repr_l2_q50", np.quantile(repr_l2, 0.5)))
    result["edge_distance_threshold"] = float(result.get("tmd_dist_q75", np.quantile(tmd_dist, 0.75)))
    result["target_distance_threshold"] = float(result.get("tmd_dist_q90", np.quantile(tmd_dist, 0.9)))
    return result


def save_calibration(scales, save_dir, filename="tmd_calibration.json"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    with open(path, "w") as f:
        json.dump(scales, f, indent=2, sort_keys=True)
    return path
