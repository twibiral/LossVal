import argparse
import os
import sys
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from opendataval.dataloader import add_gauss_noise, mix_labels, DataFetcher
from opendataval.experiment import ExperimentMediator
from tqdm import tqdm

import src.baseline.run_baselines_classification as rbc
from src.baseline.MLP_hyperparameters import CLASSIFIER_BATCH_SIZE, CLASSIFIER_MLP_HYPERPARAMETERS, \
    REGRESSION_BATCH_SIZE, REGRESSION_MLP_HYPERPARAMETERS
from src.baseline.utils import create_csv_files, get_dataset_split, mixed_noise_function
from src.LossVal.LossVal_MLP import LossVal_MLP
from src.LossVal.LossVal_evaluator import LossVal_Evaluator
from src.LossVal.loss_configurations import REGRESSION_LOSSES, CLASSIFICATION_LOSSES

# Import regression dataset file to export the newly added CTR23 datasets into the OpenDataVal benchmark:


REGRESSION_DATASETS = [
    "kin8nm",
    "white_wine",
    "cpu_activity",
    "pumadyn32nh",
    "wave_energy",
    "superconductivity"
]
CLASSIFICATION_DATASETS = [
    "2dplanes",
    "electricity",
    "MiniBooNE",
    "pol",
    "fried",
    "nomao"
]

# Reps yet:
NOISE_RATES = [0.05, 0.1, 0.15, 0.2]
NR_EPOCHS_TO_TEST = [5, 30]
LEARNING_RATES = [0.01]#[0.1, 0.01, 0.001]
NR_EXPERIMENT_REPETITIONS = 5
METRIC = {"regression": "neg_mse", "classification": "accuracy"}

MLP_HYPERPARAMETERS_REGRESSION = REGRESSION_MLP_HYPERPARAMETERS
TRAINING_HYPERPARAMETERS_REGRESSION = {"batch_size": REGRESSION_BATCH_SIZE}

MLP_HYPERPARAMETERS_CLASSIFICATION = CLASSIFIER_MLP_HYPERPARAMETERS
TRAINING_HYPERPARAMETERS_CLASSIFICATION = {"batch_size": CLASSIFIER_BATCH_SIZE}

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data_files/")
USE_ORIGINAL_DATA_SPLITTING = True

RESULTS_DIR_REGRESSION = os.path.join(os.path.dirname(__file__), "../../results/lossval/regression/")
RESULTS_DIR_CLASSIFICATION = os.path.join(os.path.dirname(__file__), "../../results/lossval/classification/")
os.makedirs(RESULTS_DIR_REGRESSION, exist_ok=True)
os.makedirs(RESULTS_DIR_CLASSIFICATION, exist_ok=True)
RESULTS_DIR = {"regression": RESULTS_DIR_REGRESSION, "classification": RESULTS_DIR_CLASSIFICATION}


def run_LossVal_experiments(device: torch.device = torch.device("cpu"),
                            mode="noisy_label_detection", nr_repetitions: int = NR_EXPERIMENT_REPETITIONS,
                            datasets=CLASSIFICATION_DATASETS, nr_epochs_to_test=NR_EPOCHS_TO_TEST):
    """
    Run the noisy label detection experiment
    """
    if mode == "noisy_label_detection":
        noise_func = mix_labels
    elif mode == "noisy_feature_detection":
        noise_func = add_gauss_noise
    elif mode == "mixed_noise_detection":
        noise_func = mixed_noise_function
    else:
        raise ValueError(f"Unknown mode '{mode}'. "
                         f"Must be either 'noisy_label_detection', 'noisy_feature_detection', "
                         f"or 'mixed_noise_detection'")

    bar_length = len(datasets) * nr_repetitions * len(NOISE_RATES)
    t_bar = tqdm(desc=mode, unit="dataset", total=bar_length, position=1)

    for dataset_name in datasets:
        if dataset_name in REGRESSION_DATASETS:
            is_classification = False
            result_dir = RESULTS_DIR["regression"]
            hyper_params = MLP_HYPERPARAMETERS_REGRESSION

        elif dataset_name in CLASSIFICATION_DATASETS:
            is_classification = True
            result_dir = RESULTS_DIR["classification"]
            hyper_params = MLP_HYPERPARAMETERS_CLASSIFICATION

        else:
            raise RuntimeError(f"Unknown dataset '{dataset_name}'.")

        task = 'classification' if is_classification else 'regression'

        train_count, valid_count, test_count = get_dataset_split(dataset_name, DATA_DIR, USE_ORIGINAL_DATA_SPLITTING)
        create_csv_files(result_dir, mode, dataset=dataset_name)
        result_dir_with_ds_name = os.path.join(result_dir, dataset_name)

        # Run the experiment for the dataset; try all noise rates and learning rates
        for noise_rate in NOISE_RATES:
            t_bar.set_description(f"{mode}: {dataset_name} (noise rate: {noise_rate}; task: {task})")

            for _ in range(nr_repetitions):
                # Prepare the data: fetch the dataset, split it, add noise
                data_fetcher = DataFetcher(dataset_name=dataset_name, cache_dir=DATA_DIR, force_download=False)
                data_fetcher = data_fetcher.split_dataset_by_count(train_count, valid_count, test_count)
                data_fetcher = data_fetcher.noisify(noise_func, noise_rate=noise_rate)

                # Prepare the model and the experiment:
                evaluators = create_LossVal_evaluators(device, is_classification, nr_epochs_to_test, LEARNING_RATES)
                model = LossVal_MLP(
                    input_dim=data_fetcher.covar_dim[0],
                    output_dim=data_fetcher.label_dim[0],
                    training_set_size=len(data_fetcher.x_train),
                    is_classification=is_classification,
                    **hyper_params
                )

                exper_med = ExperimentMediator(data_fetcher, model, metric_name=METRIC[task], train_kwargs=hyper_params)

                remove_high_low_mode = "classification" if is_classification else "regression"
                rbc.run_experiments(evaluators, exper_med, result_dir_with_ds_name, mode, dataset_name, noise_rate, model="LossVal MLP", remove_high_low_mode=remove_high_low_mode)

                plt.close('all')
                t_bar.update(1)

    print(f"Finished running {mode} experiment. Noise rates: {NOISE_RATES}; {len(datasets)} datasets.")


def create_LossVal_evaluators(device, is_classification, nr_epochs_to_test, learning_rates):
    evaluators = []
    all_loss_configurations = CLASSIFICATION_LOSSES if is_classification else REGRESSION_LOSSES

    all_loss_configurations = {k: v for k, v in all_loss_configurations.items() if "ABLATION" not in k}

    for loss_configuration in all_loss_configurations:
        for nr_epochs in nr_epochs_to_test:
            for lr in learning_rates:
                evaluators.append(LossVal_Evaluator(device, loss_configuration, nr_epochs=nr_epochs, lr=lr))

    return evaluators


def main(datasets, device, nr_repetitions, nr_epochs_to_test):
    print(" ===== RUNNING LOSSVAL DATA VALUATION ===== ")

    print("\n --- Experiment 1: Running noisy label detection --- ")
    run_LossVal_experiments(device=device, mode="noisy_label_detection", nr_repetitions=nr_repetitions,
                            datasets=datasets, nr_epochs_to_test=nr_epochs_to_test)

    print("\n --- Experiment 2: Running noisy feature detection --- ")
    run_LossVal_experiments(device=device, mode="noisy_feature_detection", nr_repetitions=nr_repetitions,
                            datasets=datasets, nr_epochs_to_test=nr_epochs_to_test)

    print("\n --- Experiment 3: Label mixing AND feature noise detection --- ")
    run_LossVal_experiments(device=device, mode="mixed_noise_detection", nr_repetitions=nr_repetitions,
                            datasets=datasets, nr_epochs_to_test=nr_epochs_to_test)


if __name__ == '__main__':
    def lower_priority():
        """Set the priority of the process to below-normal.
        https://stackoverflow.com/a/1023269/13027030"""
        try:
            sys.getwindowsversion()
        except AttributeError:
            is_windows = False
        else:
            is_windows = True

        if is_windows:
            # Based on:
            #   "Recipe 496767: Set Process Priority In Windows" on ActiveState
            #   http://code.activestate.com/recipes/496767/
            import win32api
            import win32process
            import win32con

            pid = win32api.GetCurrentProcessId()
            handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
            win32process.SetPriorityClass(handle, win32process.IDLE_PRIORITY_CLASS)  # BELOW_NORMAL_PRIORITY_CLASS
            # https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setpriorityclass
        # else:  # no need to lower the priority on linux machines
        #     import os
        #     os.nice(1)

    print(f"Current working directory: {os.getcwd()}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Parse arguments; if --dataset is set, overwrite the list of datasets
    # if --device is set, overwrite the device
    # if --nr_repetitions is set, overwrite the number of repetitions
    parser = argparse.ArgumentParser(description='Run classification baselines.')
    parser.add_argument('--nr_epochs', type=str,
                        default=NR_EPOCHS_TO_TEST,
                        help='List of dataset names to run the experiments on.')
    parser.add_argument('--device', type=str, default=device, help='Device to use for the experiments.')
    parser.add_argument('--nr_repetitions', type=int, default=NR_EXPERIMENT_REPETITIONS,
                        help='Number of repetitions for each experiment.')
    parser.add_argument('--dataset', type=str, nargs='+',
                        default=list(set(REGRESSION_DATASETS + CLASSIFICATION_DATASETS)),
                        help='List of dataset names to run the experiments on.')
    args = parser.parse_args()

    print("Number repetitions:", args.nr_repetitions)
    print("Datasets:", args.dataset)
    print("Number of epochs to test:", args.nr_epochs)
    print("Model: LossVal MLP")
    print("Device:", args.device)

    start_time = datetime.now()

    main(args.dataset, torch.device(args.device), args.nr_repetitions, args.nr_epochs)

    time_elapsed = datetime.now() - start_time
    print(f"Total runtime: {time_elapsed}")
