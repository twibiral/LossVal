import functools

from src.LossVal import loss_functions

REGRESSION_LOSSES = {
    # Main Loss:
    "Weighted MSE loss with squared Sinkhorn": functools.partial(
        loss_functions.loss_and_distance_multiplicative,
        loss_fn=loss_functions.weighted_mse,
        distance_fn=loss_functions.sinkhorn_distance_squared,
    ),
    # Without distance
    "[ABLATION] Weighted MSE": loss_functions.weighted_mse,

    # Distance only
    "[ABLATION] Sinkhorn only": lambda train_y_pred, train_y_true, train_X,
                val_X, val_y, weights, sample_ids, device: loss_functions.sinkhorn_distance(train_X, train_y_true, val_X, val_y, sample_ids, weights, device),
    "[ABLATION] Squared Sinkhorn only": lambda train_y_pred, train_y_true, train_X,
                val_X, val_y, weights, sample_ids, device: loss_functions.sinkhorn_distance_squared(train_X, train_y_true, val_X, val_y, sample_ids, weights, device),

    # Multiplicative
    "[ABLATION] Weighted MSE loss with Sinkhorn": functools.partial(
        loss_functions.loss_and_distance_multiplicative,
        loss_fn=loss_functions.weighted_mse,
        distance_fn=loss_functions.sinkhorn_distance,
    ),

    # Additive
    "[ABLATION] Weighted MSE loss with Sinkhorn (additive)": functools.partial(
        loss_functions.loss_and_distance_additive,
        loss_fn=loss_functions.weighted_mse,
        distance_fn=loss_functions.sinkhorn_distance,
    ),
    "[ABLATION] Weighted MSE loss with squared Sinkhorn (additive)": functools.partial(
        loss_functions.loss_and_distance_additive,
        loss_fn=loss_functions.weighted_mse,
        distance_fn=loss_functions.sinkhorn_distance_squared,
    ),
}

CLASSIFICATION_LOSSES = {
    # Main Loss:
    "Weighted CE loss with squared Sinkhorn": functools.partial(
        loss_functions.loss_and_distance_multiplicative,
        loss_fn=loss_functions.weighted_cross_entropy,
        distance_fn=loss_functions.sinkhorn_distance_squared,
    ),

    # Without distance
    "[ABLATION] Weighted CE": loss_functions.weighted_cross_entropy,

    # Distance only
    "[ABLATION] Sinkhorn only": lambda train_y_pred, train_y_true, train_X,
                                       val_X, val_y, weights, sample_ids, device: loss_functions.sinkhorn_distance(
        train_X, train_y_true, val_X, val_y, sample_ids, weights, device),
    "[ABLATION] Squared Sinkhorn only": lambda train_y_pred, train_y_true, train_X,
                                               val_X, val_y, weights, sample_ids,
                                               device: loss_functions.sinkhorn_distance_squared(train_X, train_y_true,
                                                                                                val_X, val_y,
                                                                                                sample_ids, weights,
                                                                                                device),

    # Multiplicative
    "[ABLATION] Weighted CE loss with Sinkhorn": functools.partial(
        loss_functions.loss_and_distance_multiplicative,
        loss_fn=loss_functions.weighted_cross_entropy,
        distance_fn=loss_functions.sinkhorn_distance,
    ),

    # Additive
    "[ABLATION] Weighted CE loss with Sinkhorn (additive)": functools.partial(
        loss_functions.loss_and_distance_additive,
        loss_fn=loss_functions.weighted_cross_entropy,
        distance_fn=loss_functions.sinkhorn_distance,
    ),
    "[ABLATION] Weighted CE loss with squared Sinkhorn (additive)": functools.partial(
        loss_functions.loss_and_distance_additive,
        loss_fn=loss_functions.weighted_cross_entropy,
        distance_fn=loss_functions.sinkhorn_distance_squared,
    ),
}
