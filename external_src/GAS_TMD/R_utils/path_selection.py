import numpy as np


def select_reachable_path_node(
    provider,
    psi_obs,
    path_embeds,
    edge_distance_threshold,
    mode="tmd_distance",
    repr_cluster_threshold=None,
):
    if len(path_embeds) == 0:
        return None, np.asarray([])
    if mode == "tmd_distance":
        dists = provider.distance_embeddings(np.asarray(psi_obs)[None], np.asarray(path_embeds))[0]
        valid = np.where(dists <= edge_distance_threshold)[0]
    else:
        dists = np.linalg.norm(np.asarray(path_embeds) - np.asarray(psi_obs), axis=-1)
        threshold = edge_distance_threshold if repr_cluster_threshold is None else repr_cluster_threshold
        valid = np.where(dists <= threshold)[0]

    if len(valid) == 0:
        return 0, dists
    return int(valid[-1]), dists
