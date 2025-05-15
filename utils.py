def extract(a, t, x_shape):
    """
    Extract values from a 1D tensor 'a' at indices 't' and reshape to x_shape-compatible.
    """
    out = a.gather(0, t).reshape(-1, 1, 1, 1)
    return out.expand(x_shape)

def get_num_groups(channels):
    for g in reversed(range(1, channels + 1)):
        if channels % g == 0:
            return min(g, 8)
    return 1