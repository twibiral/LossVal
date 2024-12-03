"""
This module contains the weighted loss functions used in LossVal.
We provide a weighted cross-entropy loss for classification tasks and a weighted MSE for regression tasks.

For the weighted distribution distance, we provide the Sinkhorn distance as a differentiable loss function.

Fixed implementations for the loss functions used in the experiments are provided.
At the end, we define more generic functions to combine any loss function with any distance measure.
"""
import torch
from geomloss import SamplesLoss


# ===== Sinkhorn Distance =====
def sinkhorn_distance(train_X: torch.Tensor, train_y_true: torch.Tensor, val_X: torch.Tensor, val_y: torch.Tensor,
                      sample_ids: torch.Tensor, weights: torch.Tensor, device: torch.device) -> torch.Tensor:
    """ Calculate the weighted Sinkhorn distance as a distribution distance measure in a differentiable way.
    The Sinkhorn distance is a fast approximation of the Wasserstein (optimal transport) distance.

    Note that this function calculates the Sinkhorn distance between the training and validation data, where only the
    training data is weighted. Moreover, we use joint distributions of input features X and target values y!

    All functions used in this function are differentiable via pytorch. This is necessary to use the Sinkhorn distance
    in a neural network training process (as part of the loss).

    :param train_y_pred: The predicted y-values of the training data
    :param train_y_true: The true y-values of the training data
    :param train_X: The training data
    :param val_X: The validation data
    :param sample_ids: The sample ids of the training data
    :param val_y: The validation y-values
    :param weights: The weights of the training data
    :param device: The device to use
    :param args: Additional arguments
    :param kwargs: Additional keyword arguments
    :return: The Sinkhorn distance
    """
    weights = weights.index_select(0, sample_ids)  # Select the weights corresponding to the sample_ids

    # Dynamically chose a good value for the diameter parameter
    dist_matrix = torch.cdist(train_X, val_X, p=2)
    max_distance = dist_matrix.max().item()

    sinkhorn_distance = SamplesLoss(loss="sinkhorn", diameter=max_distance*1.1)
    dist_loss = sinkhorn_distance(weights, train_X, torch.ones(val_X.shape[0], requires_grad=True).to(device), val_X)

    return torch.abs(dist_loss)     # Ensure that the distance is positive


def sinkhorn_distance_squared(train_X: torch.Tensor, train_y_true: torch.Tensor, val_X: torch.Tensor, val_y: torch.Tensor,
                                sample_ids: torch.Tensor, weights: torch.Tensor, device: torch.device) -> torch.Tensor:
    """ Squared variant of the Sinkhorn distance. """
    weights = weights.index_select(0, sample_ids)  # Select the weights corresponding to the sample_ids

    # Dynamically chose a good value for the diameter parameter
    dist_matrix = torch.cdist(train_X, val_X, p=2)
    max_distance = dist_matrix.max().item()

    sinkhorn_distance = SamplesLoss(loss="sinkhorn", diameter=max_distance*1.1)
    dist_loss = sinkhorn_distance(weights, train_X, torch.ones(val_X.shape[0], requires_grad=True).to(device), val_X)

    return torch.abs(dist_loss)     # Ensure that the distance is positive



# ===== Weighted Loss Functions =====
def weighted_cross_entropy(train_y_pred: torch.Tensor, train_y_true: torch.Tensor,
                           weights: torch.Tensor, sample_ids: torch.Tensor,
                           device: torch.device, epsilon_for_log=1e-8, *args, **kwargs) -> torch.Tensor:
    """ Compute a weighted cross-entropy loss where each instance is weighted separately.
    :param train_y_pred: predictions of the model on the training set
    :param train_y_true: targets of the training set
    :param weights: a vector containing a weight for each instance
    :param sample_ids: the indices of the samples that are used in this batch
    :param device: device to run the computations on
    :param epsilon_for_log: a small value to add to the log to avoid numerical instability
    :return: the loss
    """
    # Compute the cross-entropy loss; targets are already one-hot encoded!
    loss = -torch.sum(train_y_true * torch.log(train_y_pred + epsilon_for_log), dim=1)

    weights = weights.index_select(0, sample_ids)  # Select the weights corresponding to the sample_ids
    weighted_loss = torch.sum(weights @ loss)  # Loss is a vector, weights is a matrix

    return weighted_loss


def weighted_cross_entropy_with_correction(train_y_pred: torch.Tensor, train_y_true: torch.Tensor,
                           weights: torch.Tensor, sample_ids: torch.Tensor,
                           device: torch.device, epsilon_for_log=1e-8, *args, **kwargs) -> torch.Tensor:
    """ Compute a weighted cross-entropy loss where each instance is weighted separately.
    :param train_y_pred: predictions of the model on the training set
    :param train_y_true: targets of the training set
    :param weights: a vector containing a weight for each instance
    :param sample_ids: the indices of the samples that are used in this batch
    :param device: device to run the computations on
    :param epsilon_for_log: a small value to add to the log to avoid numerical instability
    :return: the loss
    """
    # Compute the cross-entropy loss; targets are already one-hot encoded!
    loss = -torch.sum(train_y_true * torch.log(train_y_pred + epsilon_for_log), dim=1)

    weights = weights.index_select(0, sample_ids)  # Select the weights corresponding to the sample_ids
    weighted_loss = torch.sum(weights @ loss)  # Loss is a vector, weights is a matrix

    return weighted_loss + torch.sum(loss)


def weighted_cross_entropy_with_correction_multiplicative(train_y_pred: torch.Tensor, train_y_true: torch.Tensor,
                           weights: torch.Tensor, sample_ids: torch.Tensor,
                           device: torch.device, epsilon_for_log=1e-8, *args, **kwargs) -> torch.Tensor:
    """ Compute a weighted cross-entropy loss where each instance is weighted separately.
    :param train_y_pred: predictions of the model on the training set
    :param train_y_true: targets of the training set
    :param weights: a vector containing a weight for each instance
    :param sample_ids: the indices of the samples that are used in this batch
    :param device: device to run the computations on
    :param epsilon_for_log: a small value to add to the log to avoid numerical instability
    :return: the loss
    """
    # Compute the cross-entropy loss; targets are already one-hot encoded!
    loss = -torch.sum(train_y_true * torch.log(train_y_pred + epsilon_for_log), dim=1)

    weights = weights.index_select(0, sample_ids)  # Select the weights corresponding to the sample_ids
    weighted_loss = torch.sum(weights @ loss)  # Loss is a vector, weights is a matrix

    return weighted_loss * torch.sum(loss)


def weighted_mse(train_y_pred: torch.Tensor, train_y_true: torch.Tensor,
                 weights: torch.Tensor, sample_ids: torch.Tensor,
                 device: torch.device, *args, **kwargs) -> torch.Tensor:
    """ Compute a weighted mean squared loss where each instance is weighted separately.
    :param train_y_pred: predictions of the model on the training set
    :param train_y_true: targets of the training set
    :param weights: a vector containing a weight for each instance
    :param sample_ids: the indices of the samples that are used in this batch
    :param device: device to run the computations on
    :param epsilon_for_log: a small value to add to the log to avoid numerical instability
    :return: the loss
    """
    # Compute the mean squared error per instance
    loss = torch.sum((train_y_true - train_y_pred) ** 2, dim=1)

    weights = weights.index_select(0, sample_ids)  # Select the weights corresponding to the sample_ids
    weighted_loss = torch.sum(weights @ loss)  # Loss is a vector, weights is a matrix

    return weighted_loss



def weighted_mse_with_correction(train_y_pred: torch.Tensor, train_y_true: torch.Tensor,
                 weights: torch.Tensor, sample_ids: torch.Tensor,
                 device: torch.device, *args, **kwargs) -> torch.Tensor:
    """ Compute a weighted mean squared loss where each instance is weighted separately.
    :param train_y_pred: predictions of the model on the training set
    :param train_y_true: targets of the training set
    :param weights: a vector containing a weight for each instance
    :param sample_ids: the indices of the samples that are used in this batch
    :param device: device to run the computations on
    :param epsilon_for_log: a small value to add to the log to avoid numerical instability
    :return: the loss
    """
    # Compute the mean squared error per instance
    loss = torch.sum((train_y_true - train_y_pred) ** 2, dim=1)

    weights = weights.index_select(0, sample_ids)  # Select the weights corresponding to the sample_ids
    weighted_loss = torch.sum(weights @ loss)  # Loss is a vector, weights is a matrix

    return weighted_loss + torch.sum(loss)


def weighted_mse_with_correction_multiplicative(train_y_pred: torch.Tensor, train_y_true: torch.Tensor,
                 weights: torch.Tensor, sample_ids: torch.Tensor,
                 device: torch.device, *args, **kwargs) -> torch.Tensor:
    """ Compute a weighted mean squared loss where each instance is weighted separately.
    :param train_y_pred: predictions of the model on the training set
    :param train_y_true: targets of the training set
    :param weights: a vector containing a weight for each instance
    :param sample_ids: the indices of the samples that are used in this batch
    :param device: device to run the computations on
    :param epsilon_for_log: a small value to add to the log to avoid numerical instability
    :return: the loss
    """
    # Compute the mean squared error per instance
    loss = torch.sum((train_y_true - train_y_pred) ** 2, dim=1)

    weights = weights.index_select(0, sample_ids)  # Select the weights corresponding to the sample_ids
    weighted_loss = torch.sum(weights @ loss)  # Loss is a vector, weights is a matrix

    return weighted_loss * torch.sum(loss)


def loss_and_distance_multiplicative(train_y_pred: torch.Tensor, train_y_true: torch.Tensor, weights: torch.Tensor,
                                     sample_ids: torch.Tensor, train_X: torch.Tensor, val_X: torch.Tensor,
                                     val_y: torch.Tensor,
                                     loss_fn, distance_fn, device: torch.device, *args, **kwargs) -> torch.Tensor:
    """
    Combine any loss function and any distance measure in a multiplicative way => loss_fn * distance_fn
    :param train_y_pred: predictions of the model on the training set
    :param train_y_true: targets of the training set
    :param weights: a vector containing a weight for each instance
    :param train_X: features of the test set
    :param val_X: features of the validation set
    :param device: device to run the computations on
    :return: the loss
    """
    loss = loss_fn(train_y_pred, train_y_true, weights, sample_ids, device)
    distribution_distance = distance_fn(train_X, train_y_true, val_X, val_y, sample_ids, weights, device)

    return loss * distribution_distance


def loss_and_distance_additive(train_y_pred: torch.Tensor, train_y_true: torch.Tensor, weights: torch.Tensor,
                               sample_ids: torch.Tensor, train_X: torch.Tensor, val_X: torch.Tensor,
                               val_y: torch.Tensor,
                               loss_fn, distance_fn, device: torch.device, alpha: float = 0.5, *args,
                               **kwargs) -> torch.Tensor:
    """
    Combine any loss function and any distance measure in an additive way, using alpha for modulation.
    => alpha * loss_fn + (1 - alpha) * distance_fn
    :param train_y_pred: predictions of the model on the training set
    :param train_y_true: targets of the training set
    :param weights: a vector containing a weight for each instance
    :param train_X: features of the test set
    :param val_X: features of the validation set
    :param device: device to run the computations on
    :return: the loss
    """
    loss = loss_fn(train_y_pred, train_y_true, weights, sample_ids, device)
    distribution_distance = distance_fn(train_X, train_y_true, val_X, val_y, sample_ids, weights, device)

    return alpha * loss + (1 - alpha) * distribution_distance


# ===== Helper functions =====
def move_all_tensors_to_device(device, *tensors):
    """ Helper to simplify moving multiple tensors to the same device. """
    return tuple(tensor.to(device) for tensor in tensors)
