from collections import OrderedDict
from typing import Optional, Callable, Union

import numpy as np
import torch
from opendataval.dataloader import CatDataset
from opendataval.model import TorchPredictMixin, TorchGradMixin, RegressionMLP, ClassifierMLP
from torch import nn
from torch.utils.data import Dataset, DataLoader


class LossVal_MLP(TorchPredictMixin, TorchGradMixin):
    """ Pytorch MLP for LossVal
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
        self.track_weights = track_weights
        self.weight_history = None

    def forward(self, x):
        x = self.mlp(x)
        return x

    def get_data_weights(self) -> np.ndarray:
        return self.data_weights.detach().cpu().numpy().copy()

    def fit(self,
            x_train: Union[torch.Tensor, Dataset],
            y_train: Union[torch.Tensor, Dataset],
            sample_weight: Optional[torch.Tensor] = None,
            batch_size: int = 32,
            epochs: int = 1,
            lr: float = 0.01,
            val_X: torch.Tensor = None, val_y: torch.Tensor = None,
            loss_function: Callable = None, verbose: bool = False, **kwargs):
        """Fits the model on the training data.

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
        if val_X is None or val_y is None:  # This is necessary to enable the data addition and removal experiments!
            print("Warning: No validation data provided! Assuming Data Removal experiment and training "
                  "without validation.")
            if self.is_classification:
                return ClassifierMLP(
                    input_dim=self.input_dim,
                    num_classes=self.output_dim,
                    layers=self.nr_layers,
                    hidden_dim=self.hidden_dim,
                    act_fn=self.act_fn
                ).fit(x_train, y_train, sample_weight, batch_size, epochs, lr)
            else:
                return RegressionMLP(
                    input_dim=self.input_dim,
                    num_classes=self.output_dim,
                    layers=self.nr_layers,
                    hidden_dim=self.hidden_dim,
                    act_fn=self.act_fn
                ).fit(x_train, y_train, sample_weight, batch_size, epochs, lr)

        assert loss_function is not None, "Loss function must be provided!"
        assert sample_weight is None, "Sample weights are not supported for this method."

        def move_dataset_to_device(dataset_, device_):
            data_loader = DataLoader(dataset=dataset_, batch_size=len(dataset_), shuffle=False)

            # Extract the full batch from DataLoader and put data on the device
            sample_ids_, x_data, targets = next(iter(data_loader))
            sample_ids_ = sample_ids_.to(device_)
            x_data = x_data.to(device_)
            targets = targets.to(device_)

            return torch.utils.data.TensorDataset(sample_ids_, x_data, targets)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        indices = torch.arange(len(x_train))
        dataset = CatDataset(indices, x_train, y_train)
        val_X, val_y = val_X.to(self.device), val_y.to(self.device)

        # Already load the data on the device; the datasets are small enough
        dataset = move_dataset_to_device(dataset, self.device)
        dataloader = DataLoader(dataset, batch_size, shuffle=True)

        self.train()
        # weight_history = [self.get_data_weights()]
        for _ in range(int(epochs)):
            for (sample_ids, x_batch, y_batch) in dataloader:
                optimizer.zero_grad()
                y_hat = self.__call__(x_batch)

                # Here the modified loss is called.
                loss = loss_function(train_y_pred=y_hat, train_y_true=y_batch, train_X=x_batch, val_X=val_X, val_y=val_y,
                                         weights=self.data_weights, sample_ids=sample_ids, device=self.device)

                loss.backward()
                optimizer.step()  # Important: This step also updates the sample weights (the data valuation)

            # if self.track_weights:
            #     weight_history.append(self.get_data_weights())

        # if self.track_weights:  # Save the weight history
        #     self.weight_history = weight_history
        return self
