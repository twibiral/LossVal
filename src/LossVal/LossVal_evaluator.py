from typing import Callable, Union

import numpy as np
import torch
from opendataval.dataval import DataEvaluator, ModelMixin
from opendataval.model import GradientModel
from opendataval.model.api import TorchModel
from torch import nn

from src.LossVal import loss_configurations
from src.LossVal.LossVal_MLP import LossVal_MLP


class LossVal_Evaluator(DataEvaluator, ModelMixin):
    def __init__(self, device,  loss_function: Union[str, Callable], nr_epochs=1, lr=None, *args, **kwargs):
        super(LossVal_Evaluator, self).__init__()
        self.args = args
        self.kwargs = kwargs
        self.device = device

        self.nr_epochs = nr_epochs
        self.lr = lr

        self.pred_model = None

        self.loss_function_str = str(loss_function)
        if isinstance(loss_function, str):
            if loss_function in loss_configurations.REGRESSION_LOSSES:
                self.criterion = loss_configurations.REGRESSION_LOSSES[loss_function]
            elif loss_function in loss_configurations.CLASSIFICATION_LOSSES:
                self.criterion = loss_configurations.CLASSIFICATION_LOSSES[loss_function]
            else:
                raise ValueError(f"Loss '{loss_function}' not found in loss configurations!")

        else:
            self.criterion = loss_function

    def input_model(self, pred_model: GradientModel):
        """Input the prediction model with gradient.

        Parameters
        ----------
        pred_model : GradientModel
            Prediction model with a gradient
        """
        assert (isinstance(pred_model, LossVal_MLP)), ("The Evaluator for LossVal only works with the LossValMLP!")

        self.pred_model = pred_model.clone().to(self.device)

        return self

    def input_data(self, x_train: torch.Tensor, y_train: torch.Tensor, x_valid: torch.Tensor, y_valid: torch.Tensor):
        """Store and transform input data

        Parameters
        ----------
        x_train : torch.Tensor
            Data covariates
        y_train : torch.Tensor
            Data labels
        x_valid : torch.Tensor
            Test+Held-out covariates
        y_valid : torch.Tensor
            Test+Held-out labels
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid

        self.num_points = len(x_train)
        return self

    def train_data_values(self, *args, **kwargs):
        """Calculate the data values.

        Parameters
        ----------
        args : tuple[Any], optional
            Training positional args
        kwargs : dict[str, Any], optional
            Training key word arguments
        """
        # Check that the model is a torch model:
        assert isinstance(self.pred_model, TorchModel)
        assert isinstance(self.pred_model, nn.Module)
        assert isinstance(self.pred_model, LossVal_MLP)
        self.pred_model: LossVal_MLP

        # Move copy of all data to the device
        self.x_train = self.x_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.x_valid = self.x_valid.to(self.device)
        self.y_valid = self.y_valid.to(self.device)

        # Train the model
        self.pred_model = self.pred_model.to(self.device)

        if "epochs" in kwargs:  # Just in case there are conflicting parameters (when one is passed by the ExperimentMediator)
            kwargs.pop("epochs")

        if self.lr is None:
            self.pred_model.fit(self.x_train, self.y_train, epochs=self.nr_epochs, loss_function=self.criterion,
                            val_X=self.x_valid, val_y=self.y_valid, *args, **kwargs)
        else:
            self.pred_model.fit(self.x_train, self.y_train, epochs=self.nr_epochs, lr=self.lr, loss_function=self.criterion,
                            val_X=self.x_valid, val_y=self.y_valid, *args, **kwargs)

        return self

    def evaluate_data_values(self):
        """Return data values for each training data point.

        Returns
        -------
        np.ndarray
            Predicted data values/selection for every training data point
        """
        return self.pred_model.get_data_weights()

    def __repr__(self):
        lr_str = "" if self.lr is None else f", lr={self.lr}"
        return f"LossVal_Evaluator(loss={self.loss_function_str}, nr_epochs={self.nr_epochs}{lr_str})"

    def __str__(self):
        return self.__repr__()
