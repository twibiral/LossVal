import argparse
import os
from datetime import datetime

import torch
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from src.baseline.MLP_hyperparameters import REGRESSION_BATCH_SIZE, REGRESSION_LEARNING_RATE, REGRESSION_MLP_HYPERPARAMETERS
from src.baseline.run_baselines_classification import run_baseline_experiments
from src.baseline.utils import instantiate_evaluators

# Import regression dataset file to export the newly added CTR23 datasets into the OpenDataVal benchmark:
import src.baseline.regression_datasets

# Use datasets from OpenML-CTR23 => https://www.openml.org/search?type=benchmark&sort=tasks_included&study_type=task&id=353
ALL_DATASETS = ["kin8nm", "white_wine", "cpu_activity", "pumadyn32nh", "wave_energy", "superconductivity"]

NOISE_RATES = [0.05, 0.1, 0.15, 0.2]
NR_EXPERIMENT_REPETITIONS = 5
METRIC = "neg_mse"
NR_TRAINING_EPOCHS = 5

REGRESSION_TRAIN_ARGS = {'epochs': NR_TRAINING_EPOCHS,
                         'batch_size': REGRESSION_BATCH_SIZE,
                         'lr': REGRESSION_LEARNING_RATE}

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data_files/")
USE_ORIGINAL_DATA_SPLITTING = True

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results/baselines/regression/")
os.makedirs(RESULTS_DIR, exist_ok=True)     # Create the results directory if it does not exist


@ignore_warnings(category=ConvergenceWarning)   # Sometimes AME doesn't converge well. More iterations could fix the warning, but not the underlying problem.
def main(datasets, model, device, nr_repetitions):
    print(" ===== RUNNING BASELINES (Regression) ===== ")
    print("Model:", model)
    print("Device:", device)
    print("Evaluators:", instantiate_evaluators(validation_set_size=100))
    print("Datasets:", datasets)

    print("\n --- Experiment 1: Running noisy label detection --- ")
    run_baseline_experiments(model, device=device, mode="noisy_label_detection", result_dir=RESULTS_DIR,
                             nr_repetitions=nr_repetitions, datasets=datasets, train_kwargs=REGRESSION_TRAIN_ARGS,
                             remove_high_low_mode="regression", model_hyperparameters=REGRESSION_MLP_HYPERPARAMETERS)

    print("\n --- Experiment 2: Running noisy feature detection --- ")
    run_baseline_experiments(model, device=device, mode="noisy_feature_detection", result_dir=RESULTS_DIR,
                             nr_repetitions=nr_repetitions, datasets=datasets, train_kwargs=REGRESSION_TRAIN_ARGS,
                             remove_high_low_mode="regression", model_hyperparameters=REGRESSION_MLP_HYPERPARAMETERS)

    print("\n --- Experiment 3: Label mixing AND feature noise detection --- ")
    run_baseline_experiments(model, device=device, mode="mixed_noise_detection", result_dir=RESULTS_DIR,
                             nr_repetitions=nr_repetitions, datasets=datasets, train_kwargs=REGRESSION_TRAIN_ARGS,
                             remove_high_low_mode="regression", model_hyperparameters=REGRESSION_MLP_HYPERPARAMETERS)

if __name__ == '__main__':
    # Print current working dir:
    print(f"Current working directory: {os.getcwd()}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model = "sklinreg"
    model = "regressionmlp"

    # Parse arguments; if --dataset is set, overwrite the list of datasets
    # if --model is set, overwrite the model
    # if --device is set, overwrite the device
    # if --nr_repetitions is set, overwrite the number of repetitions
    parser = argparse.ArgumentParser(description='Run classification baselines.')
    parser.add_argument('--dataset', type=str, nargs='+', default=ALL_DATASETS, help='List of dataset names to run the experiments on.')
    parser.add_argument('--model', type=str, default=model, help='Model to use for the experiments.')
    parser.add_argument('--device', type=str, default=device, help='Device to use for the experiments.')
    parser.add_argument('--nr_repetitions', type=int, default=NR_EXPERIMENT_REPETITIONS, help='Number of repetitions for each experiment.')
    args = parser.parse_args()

    print("Number repetitions:", args.nr_repetitions)
    print("Datasets:", args.dataset)
    print("Model:", args.model)
    print("Device:", args.device)

    start_time = datetime.now()

    main(args.dataset, args.model, torch.device(args.device), args.nr_repetitions)

    time_elapsed = datetime.now() - start_time
    print(f"Total runtime: {time_elapsed}")
