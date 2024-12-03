from torch import nn

# ===== HYPERPARAMETERS FOR REGRESSION MLP =====
REGRESSION_MLP_HYPERPARAMETERS = {     # Hyperparameters for the MLP model (optimized in MLP_hyperparameter_tuning.ipynb)
    "layers": 3,
    "hidden_dim": 90,
    "act_fn": nn.Tanh(),
}
REGRESSION_LEARNING_RATE = 0.01    # Optimized like the other hyperparameters
REGRESSION_BATCH_SIZE = 32         # Optimized like the other hyperparameters


# ===== HYPERPARAMETERS FOR CLASSIFICATION MLP =====
CLASSIFIER_MLP_HYPERPARAMETERS = {     # Hyperparameters for the MLP model (optimized in MLP_hyperparameter_tuning.ipynb)
    "layers": 5,
    "hidden_dim": 100,
    "act_fn": nn.ReLU(),
}
CLASSIFIER_LEARNING_RATE = 0.1     # Optimized like the other hyperparameters
CLASSIFIER_BATCH_SIZE = 128        # Optimized like the other hyperparameters
