def extract(a, t, x_shape):
    """
    Extract values from a 1D tensor 'a' at indices 't' and reshape to x_shape-compatible.
    """
    out = a.gather(0, t).reshape(-1, 1, 1, 1)
    return out.expand(x_shape)