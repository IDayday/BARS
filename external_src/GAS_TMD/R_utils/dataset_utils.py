import numpy as np


def limit_dataset_to_complete_prefix(dataset, max_dataset_states):
    """Return a prefix ending at a terminal, or the original dataset if unset."""
    if max_dataset_states is None:
        return dataset
    max_dataset_states = int(max_dataset_states)
    if max_dataset_states <= 0 or max_dataset_states >= dataset.size:
        return dataset

    terminals = np.asarray(dataset["terminals"])
    terminal_locs = np.flatnonzero(terminals > 0)
    valid_terms = terminal_locs[terminal_locs < max_dataset_states]
    if len(valid_terms) == 0:
        raise ValueError(f"No complete trajectory found before max_dataset_states={max_dataset_states}.")
    end = int(valid_terms[-1])
    subset = {}
    for key, value in dataset.items():
        try:
            if len(value) == dataset.size:
                subset[key] = value[: end + 1]
            else:
                subset[key] = value
        except TypeError:
            subset[key] = value
    return dataset.__class__.create(**subset)
