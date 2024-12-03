import argparse
import functools
import os
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from opendataval.dataloader import add_gauss_noise, mix_labels, DataFetcher
from opendataval.experiment import ExperimentMediator, remove_high_low, discover_corrupted_sample, noisy_detection
from opendataval.metrics import Metrics
from opendataval.model import ModelFactory, ClassifierMLP, RegressionMLP
from tqdm import tqdm

from src.baseline.MLP_hyperparameters import CLASSIFIER_MLP_HYPERPARAMETERS, REGRESSION_MLP_HYPERPARAMETERS, \
    CLASSIFIER_BATCH_SIZE, CLASSIFIER_LEARNING_RATE, REGRESSION_BATCH_SIZE, REGRESSION_LEARNING_RATE
from src.baseline.utils import create_csv_files, get_dataset_split, instantiate_evaluators, mixed_noise_function

# Any dataset registered with the OpenDataVal library can be used here.
# We use the same datasets as in the OpenDataVal paper.
ALL_DATASETS = ["2dplanes", "electricity", "MiniBooNE", "pol", "fried", "nomao",]
                # "bbc-embeddings", "IMDB-embeddings", "CIFAR10-embeddings"]    # Used in the OpenDataVal paper

NOISE_RATES = [0.05, 0.1, 0.15, 0.2]
NR_EXPERIMENT_REPETITIONS = 5
METRIC = "accuracy"
NR_TRAINING_EPOCHS = 5

CLASSIFIER_TRAIN_ARGS = {'epochs': NR_TRAINING_EPOCHS,
                         'batch_size': CLASSIFIER_BATCH_SIZE,
                         'lr': CLASSIFIER_LEARNING_RATE}

DATA_DIR = os.path.join(os.path.dirname(__file__), "../../data_files/")
USE_ORIGINAL_DATA_SPLITTING = True

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "../../results/baselines/classification/")
os.makedirs(RESULTS_DIR, exist_ok=True)     # Create the results directory if it does not exist


def run_baseline_experiments(model, result_dir: str, model_hyperparameters, train_kwargs=None, device: torch.device = torch.device("cpu"),
                             mode="noisy_label_detection", nr_repetitions: int = 10, datasets=ALL_DATASETS, remove_high_low_mode="classification"):
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
        train_count, valid_count, test_count = get_dataset_split(dataset_name, DATA_DIR, USE_ORIGINAL_DATA_SPLITTING)
        create_csv_files(result_dir, mode, dataset=dataset_name)
        result_dir_with_ds_name = os.path.join(result_dir, dataset_name)

        for noise_rate in NOISE_RATES:
            t_bar.set_description(f"{mode}: {dataset_name} (noise rate: {noise_rate})")

            for _ in range(nr_repetitions):
                # Prepare the data: fetch the dataset, split it, add noise
                data_fetcher = DataFetcher(dataset_name=dataset_name, cache_dir=DATA_DIR, force_download=False)
                data_fetcher = data_fetcher.split_dataset_by_count(train_count, valid_count, test_count)
                data_fetcher = data_fetcher.noisify(noise_func, noise_rate=noise_rate)

                # Prepare the model and the experiment:
                evaluators = instantiate_evaluators(validation_set_size=valid_count)
                model_temp = ModelFactory(model, fetcher=data_fetcher, device=device, **model_hyperparameters) if isinstance(model, str) else model
                exper_med = ExperimentMediator(data_fetcher, model_temp, metric_name=METRIC, train_kwargs=train_kwargs)

                # For trying Data-OOB without MLP:
                # model_temp = ModelFactory(model, fetcher=data_fetcher,#                           device=device)  # ModelFactory(model, fetcher=data_fetcher, device=device, **model_hyperparameters) if isinstance(model, str) else model
                # exper_med = ExperimentMediator(data_fetcher, model_temp,#                                metric_name=METRIC)  # , train_kwargs=train_kwargs)
                # model_str = model
                # if train_kwargs is not None:    # To disambiguate results from the same model but different hyperparam.
                #     model_str += " (" + ", ".join(k + "=" + str(v) for k, v in train_kwargs.items()) + ")"

                model_str = model

                run_experiments(evaluators, exper_med, result_dir_with_ds_name, mode, dataset_name, noise_rate, model_str, remove_high_low_mode)

                plt.close('all')
                t_bar.update(1)

    print(f"Finished running {mode} experiment. Noise rates: {NOISE_RATES}; {len(datasets)} datasets.")


def run_experiments(evaluators: list, exper_med: ExperimentMediator, result_path, mode, dataset_name, noise_rate,
                    model, remove_high_low_mode="classification"):
    """ Run the experiments and save the results to the csv files. """
    def pad_the_df(df, model_=model):
        """ All result datasets need those additional columns. """
        df["method"] = df.index
        df["dataset"] = dataset_name
        df["noise_rate"] = noise_rate
        df["model"] = model_
        return df.reset_index(drop=True)

    # Calculate data values per data point for each evaluator:
    data_values = exper_med.compute_data_values(evaluators)  # Calculate data values for each evaluator

    # Detecting corrupted samples
    discov_corr_samples_result, _ = data_values.plot(discover_corrupted_sample)
    discov_corr_samples_result = pad_the_df(discov_corr_samples_result)
    discov_corr_samples_result.to_csv(os.path.join(result_path, f"{mode}_corruption_discovery_results.csv"),
                                      index=False, mode='a', header=False)
    # F1 scores for noisy label detection
    f1_scores = data_values.evaluate(noisy_detection).sort_values("kmeans_f1", ascending=False)
    f1_scores = pad_the_df(f1_scores)
    f1_scores.to_csv(os.path.join(result_path, f"{mode}_f1_scores.csv"), index=False, mode='a', header=False)

    # Point addition and deletion experiment
    if remove_high_low_mode == "classification":
        # Instantiate the model with the same hyperparameters as the MLP model
        # rhl_model = ClassifierMLP(input_dim=exper_med.fetcher.covar_dim[0],
        #                           num_classes=exper_med.fetcher.label_dim[0],
        #                           hidden_dim=CLASSIFIER_MLP_HYPERPARAMETERS["hidden_dim"],
        #                           layers=CLASSIFIER_MLP_HYPERPARAMETERS["layers"],
        #                           act_fn=CLASSIFIER_MLP_HYPERPARAMETERS["act_fn"])
        # model_name_ = "ClassifierMLP"
        rhl_model = ModelFactory("logisticregression", fetcher=exper_med.fetcher)
        model_name_ = "logistic_regression"
        rhl_func = functools.partial(remove_high_low, metric=Metrics.ACCURACY, model=rhl_model, percentile=0.05,
                                     # train_kwargs=CLASSIFIER_TRAIN_ARGS
                                     )

    elif remove_high_low_mode == "regression":
        # rhl_model = RegressionMLP(input_dim=exper_med.fetcher.covar_dim[0],
        #                           num_classes=exper_med.fetcher.label_dim[0],
        #                           hidden_dim=REGRESSION_MLP_HYPERPARAMETERS["hidden_dim"],
        #                           layers=REGRESSION_MLP_HYPERPARAMETERS["layers"],
        #                           act_fn=REGRESSION_MLP_HYPERPARAMETERS["act_fn"])
        # model_name_ = "RegressionMLP"
        rhl_model = ModelFactory("sklinreg", fetcher=exper_med.fetcher)
        model_name_ = "linear_regression"
        rhl_func = functools.partial(remove_high_low, metric=Metrics.NEG_MSE, model=rhl_model, percentile=0.05,
                                     # train_kwargs={'epochs': NR_TRAINING_EPOCHS,
                                     #             'batch_size': REGRESSION_BATCH_SIZE,
                                     #             'lr': REGRESSION_LEARNING_RATE}
                                     )

    else:
        raise ValueError(f"Unknown remove_high_low_mode '{remove_high_low_mode}'. "
                         f"Must be either 'classification' or 'regression'")

    rem_high_low_results, _ = data_values.plot(rhl_func)
    rem_high_low_results = pad_the_df(rem_high_low_results, model_name_)
    rem_high_low_results.to_csv(os.path.join(result_path, f"{mode}_remove_high_low_results.csv"),
                                index=False, mode='a', header=False)

    # Point addition and deletion experiment
    # rem_high_low_results, _ = data_values.plot(remove_high_low)
    # rem_high_low_results = pad_the_df(rem_high_low_results)
    # rem_high_low_results.to_csv(os.path.join(result_path, f"{mode}_remove_high_low_results.csv"),
    #                             index=False, mode='a', header=False)


def main(datasets, model, device, nr_repetitions):
    print(" ===== RUNNING BASELINES ===== ")
    print("Model:", model)
    print("Device:", device)
    print("Evaluators:", instantiate_evaluators(validation_set_size=100))
    print("Datasets:", datasets)


    print("\n --- Experiment 1: Running noisy label detection --- ")
    run_baseline_experiments(model, device=device, mode="noisy_label_detection", result_dir=RESULTS_DIR,
                             nr_repetitions=nr_repetitions, datasets=datasets, train_kwargs=CLASSIFIER_TRAIN_ARGS,
                             model_hyperparameters=CLASSIFIER_MLP_HYPERPARAMETERS)

    print("\n --- Experiment 2: Running noisy feature detection --- ")
    run_baseline_experiments(model, device=device, mode="noisy_feature_detection", result_dir=RESULTS_DIR,
                             nr_repetitions=nr_repetitions, datasets=datasets, train_kwargs=CLASSIFIER_TRAIN_ARGS,
                             model_hyperparameters=CLASSIFIER_MLP_HYPERPARAMETERS)

    print("\n --- Experiment 3: Label mixing AND feature noise detection --- ")
    run_baseline_experiments(model, device=device, mode="mixed_noise_detection", result_dir=RESULTS_DIR,
                             nr_repetitions=nr_repetitions, datasets=datasets, train_kwargs=CLASSIFIER_TRAIN_ARGS,
                             model_hyperparameters=CLASSIFIER_MLP_HYPERPARAMETERS)


if __name__ == '__main__':
    # Print current working dir:
    print(f"Current working directory: {os.getcwd()}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = "classifiermlp"     # Standard pytorch MLP => will be modified to train for 5 epochs with batch size 50
    # model = "logisticregression"     # Standard pytorch MLP => will be modified to train for 5 epochs with batch size 50

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
    print("Model hyperparameters:", CLASSIFIER_MLP_HYPERPARAMETERS)
    print("Training hyperparameters:", {'epochs': NR_TRAINING_EPOCHS, 'batch_size': CLASSIFIER_BATCH_SIZE})

    start_time = datetime.now()

    main(args.dataset, args.model, torch.device(args.device), args.nr_repetitions)

    time_elapsed = datetime.now() - start_time
    print(f"Total runtime: {time_elapsed}")
