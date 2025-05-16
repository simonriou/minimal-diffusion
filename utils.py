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

def perceptual_loss(x, y, vgg_model):
    # Assuming input in [-1, 1] -> rescale to [0, 1] for VGG
    x = (x + 1) / 2
    y = (y + 1) / 2
    x_features = vgg_model(x)
    y_features = vgg_model(y)

    loss = 0

    for xf, yf in zip(x_features, y_features):
        loss += F.mse_loss(xf, yf)
    
    return loss