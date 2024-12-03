"""
This is a standalone example PyTorch implementation of the LossVal method using PyTorch.
The LossValMLP model supports both regression and classification tasks.

You can instantiate a LossValMLP model and fit it on your data using the fit method. Note that you need to pass the
size of the training set to the model, as well as if you are doing classification or regression.
You can set the number of layers, hidden dimensions, and activation function for the model.

You can pass the loss you want to use to the fit function. Reference implementations for LossVal cross-entropy and
mean squared error are provided.

Feel free to use this code as a starting point for your own implementation. You can, for example, use a different
loss function or modify the model architecture to suit your needs.

You can find more loss implementations (that we used for the Ablation study) in `src/loss_configurations.py`.
"""


from collections import OrderedDict
from typing import Optional, Callable, Union, Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from geomloss import SamplesLoss


def LossVal_mse(train_X: torch.Tensor, train_y_true: torch.Tensor, train_y_pred: torch.Tensor,
                val_X: torch.Tensor, val_y: torch.Tensor, sample_ids: torch.Tensor,
                weights: torch.Tensor, device: torch.device) -> torch.Tensor:
    """ LossVal for regression using mean squared error.
    Give the indices of the samples in the batch to the function!
    This is necessary to select the correct subset of the weights.

    :param train_X: training data
    :param train_y_true: true labels of the training data
    :param train_y_pred: predicted labels of the training data
    :param val_X: validation data
    :param val_y: true labels of the validation data
    :param sample_ids: indices of the samples that are used in this batch
    :param weights: a vector containing a weight for each instance
    :param device: device to run the computations on
    :param epsilon_for_log: a small value to add to the log to avoid numerical instability
    :return: the LossVal loss
    """
    weights = weights.index_select(0, sample_ids)  # Select the weights corresponding to the sample_ids

    # Step 1: Compute the weighted mse loss
    loss = torch.sum((train_y_true - train_y_pred) ** 2, dim=1)
    weighted_loss = torch.sum(weights @ loss)  # Loss is a vector, weights is a matrix

    # Step 2: Compute the Sinkhorn distance between the training and validation distributions
    # Dynamically chose a good value for the diameter parameter
    dist_matrix = torch.cdist(train_X, val_X, p=2)
    max_distance = dist_matrix.max().item()

    sinkhorn_distance = SamplesLoss(loss="sinkhorn", diameter=max_distance*1.1)
    dist_loss = sinkhorn_distance(weights, train_X, torch.ones(val_X.shape[0], requires_grad=True).to(device), val_X)

    # Step 3: Combine cross entropy and Sinkhorn distance
    return weighted_loss * dist_loss**2


def LossVal_cross_entropy(train_X: torch.Tensor, train_y_true: torch.Tensor, train_y_pred: torch.Tensor,
                          val_X: torch.Tensor, val_y: torch.Tensor, sample_ids: torch.Tensor,
                          weights: torch.Tensor, device: torch.device, epsilon_for_log=1e-8) -> torch.Tensor:
    """ LossVal for classification using cross-entropy loss.
    Give the indices of the samples in the batch to the function!
    This is necessary to select the correct subset of the weights.

    :param train_X: training data
    :param train_y_true: true labels of the training data
    :param train_y_pred: predicted labels of the training data
    :param val_X: validation data
    :param val_y: true labels of the validation data
    :param sample_ids: indices of the samples that are used in this batch
    :param weights: a vector containing a weight for each instance
    :param device: device to run the computations on
    :param epsilon_for_log: a small value to add to the log to avoid numerical instability
    :return: the LossVal loss
    """
    weights = weights.index_select(0, sample_ids)  # Select the weights corresponding to the sample_ids

    # Step 1: Compute the weighted cross-entropy loss; targets are already one-hot encoded!
    loss = -torch.sum(train_y_true * torch.log(train_y_pred + epsilon_for_log), dim=1)
    weighted_loss = torch.sum(weights @ loss)  # Loss is a vector, weights is a matrix

    # Step 2: Compute the Sinkhorn distance between the training and validation distributions
    # Dynamically chose a good value for the diameter parameter
    dist_matrix = torch.cdist(train_X, val_X, p=2)
    max_distance = dist_matrix.max().item()

    sinkhorn_distance = SamplesLoss(loss="sinkhorn", diameter=max_distance*1.1)
    dist_loss = sinkhorn_distance(weights, train_X, torch.ones(val_X.shape[0], requires_grad=True).to(device), val_X)

    # Step 3: Combine cross entropy and Sinkhorn distance
    return weighted_loss * dist_loss**2


class CatDataset(Dataset[tuple[Dataset, ...]]):
    """Data set wrapping indexable Datasets. """
    def __init__(self, *datasets: list[Dataset[Any]]):
        self.datasets = [ds for ds in datasets if ds is not None]
        if not all(len(datasets[0]) == len(ds) for ds in self.datasets):
            raise ValueError("Size mismatch between data sets")

    def __getitem__(self, index) -> tuple[Any, ...]:
        """Return tuple of indexed element or tensor value on first axis."""
        return tuple(ds[index] for ds in self.datasets)

    def __len__(self) -> int:
        return len(self.datasets[0])


class LossValMLP(torch.nn.Module):
    """ Pytorch MLP for LossVal.
    Can be used for both regression and classification.
    """

    def __init__(self, input_dim: int, output_dim: int, training_set_size: int, is_classification: bool,
                 layers: int = 5, hidden_dim: int = 25, act_fn: Optional[Callable] = None, track_weights: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.training_set_size = training_set_size
        self.nr_layers = layers
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn

        # Initialize the data weights with ones
        self.data_weights = nn.Parameter(torch.ones(training_set_size), requires_grad=True)

        self.layers = layers
        self.hidden_dim = hidden_dim
        self.act_fn = act_fn

        act_fn = nn.ReLU() if act_fn is None else act_fn
        self.output_dim = output_dim

        mlp_layers = OrderedDict()
        mlp_layers["input"] = nn.Linear(input_dim, hidden_dim)
        mlp_layers["input_acti"] = act_fn

        for i in range(int(layers - 2)):
            mlp_layers[f"{i + 1}_lin"] = nn.Linear(hidden_dim, hidden_dim)
            mlp_layers[f"{i + 1}_acti"] = act_fn

        if is_classification:
            mlp_layers[f"{i + 1}_out_lin"] = nn.Linear(hidden_dim, output_dim)
            mlp_layers["output"] = nn.Softmax(-1)
        else:
            mlp_layers["output"] = nn.Linear(hidden_dim, output_dim)

        self.mlp = nn.Sequential(mlp_layers)

        self.is_classification = is_classification
        self.device = torch.device("cpu")

    def forward(self, x):
        x = self.mlp(x)
        return x

    def get_importance_scores(self) -> np.ndarray:
        return self.data_weights.detach().cpu().numpy().copy()

    def fit(self,
            x_train: Union[torch.Tensor, Dataset],
            y_train: Union[torch.Tensor, Dataset],
            sample_weight: Optional[torch.Tensor] = None,
            batch_size: int = 32,
            epochs: int = 1,
            lr: float = 0.01,
            val_X: torch.Tensor = None, val_y: torch.Tensor = None,
            loss_function: Callable = None):
        """Fits the model on the training data.
        For classification, the labels must be one-hot-encoded!

        Parameters
        ----------
        x_train : torch.Tensor | Dataset
            Data covariates
        y_train : torch.Tensor | Dataset
            Data labels
        batch_size : int, optional
            Training batch size, by default 32
        epochs : int, optional
            Number of training epochs, by default 1
        sample_weight : torch.Tensor, optional
            Weights associated with each data point, by default None
        lr : float, optional
            Learning rate for the Model, by default 0.01
        """
        assert loss_function is not None, "Loss function must be provided!"
        assert sample_weight is None, "Sample weights are not supported for this method."
        assert isinstance(x_train, torch.Tensor), "Only torch.Tensor datasets are supported."
        assert val_X is not None and val_y is not None, "Validation data must be provided!"

        def move_dataset_to_device(dataset_, device_):
            data_loader = DataLoader(dataset=dataset_, batch_size=len(dataset_), shuffle=False)

            # Extract the full batch from DataLoader and put data on the device
            sample_ids_, x_data, targets = next(iter(data_loader))
            sample_ids_ = sample_ids_.to(device_)
            x_data = x_data.to(device_)
            targets = targets.to(device_)

            return torch.utils.data.TensorDataset(sample_ids_, x_data, targets)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Make sure that the classification labels are one-hot encoded!
        indices = torch.arange(len(x_train))
        dataset = TensorDataset(indices, x_train, y_train)

        val_X, val_y = val_X.to(self.device), val_y.to(self.device)

        # Already load the data on the device; (make sure the datasets are small enough or disable this)
        dataset = move_dataset_to_device(dataset, self.device)
        dataloader = DataLoader(dataset, batch_size, shuffle=True)

        self.train()
        for _ in range(int(epochs)):
            for (sample_ids, x_batch, y_batch) in dataloader:   # Iterates batches with indices
                optimizer.zero_grad()
                y_hat = self.__call__(x_batch)

                # Here the modified loss is called.
                loss = loss_function(train_y_pred=y_hat, train_y_true=y_batch, train_X=x_batch,
                                     val_X=val_X, val_y=val_y,
                                     weights=self.data_weights, sample_ids=sample_ids, device=self.device)

                loss.backward()
                optimizer.step()  # Important: This step also updates the sample weights (the data valuation)

        return self

    def predict(self, x: Union[torch.Tensor, Dataset]) -> torch.Tensor:
        """Predict output from input tensor/data set.

        Parameters
        ----------
        x : torch.Tensor
            Input covariates

        Returns
        -------
        torch.Tensor
            Predicted tensor output
        """
        if isinstance(x, Dataset):  # Load to tensor if dataset
            x = next(iter(DataLoader(x, batch_size=len(x), pin_memory=True)))
        x = x.to(device=self.device)

        self.eval()
        with torch.no_grad():
            y_hat = self.__call__(x)

        return y_hat


# Example usage:
if __name__ == '__main__':
    # Load some iris data
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder

    iris = load_iris()
    X, y = iris.data, iris.target
    y = OneHotEncoder(sparse_output=False).fit_transform(y.reshape(-1, 1))
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train a LossValMLP model
    model = LossValMLP(
        hidden_dim=25, layers=3, act_fn=nn.ReLU(), input_dim=X_train.shape[1], output_dim=y.shape[1],
        is_classification=True, training_set_size=y_train.shape[0]
    )

    model.fit(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32),
              batch_size=30, epochs=10, lr=0.01,
              val_X=torch.tensor(X_val, dtype=torch.float32), val_y=torch.tensor(y_val, dtype=torch.float32),
              loss_function=LossVal_cross_entropy)

    # Predict on the validation set
    y_pred = model.predict(torch.tensor(X_val, dtype=torch.float32)).numpy()
    print(f"Accuracy: {np.mean((y_pred.argmax(-1) == y_val.argmax(-1)))*100.:.2f}%")

    # Visualize the importance scores
    import matplotlib.pyplot as plt
    plt.plot(np.sort(model.get_importance_scores()))
    plt.show()

    plt.hist(model.get_importance_scores(), bins=20)
    plt.show()
