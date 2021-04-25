def _permute(t, m):
    """
    Perform permutation of tuple according to mapping m
    """
    r = list(t)
    for from_idx, to_idx in m:
        r[to_idx] = t[from_idx]
    return r