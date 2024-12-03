import os
import random

import numpy as np
import pandas as pd
import torch
from opendataval.dataloader import DataFetcher
from opendataval.dataloader.util import IndexTransformDataset
from opendataval.dataval import DataOob, RandomEvaluator, LavaEvaluator, LeaveOneOut, AME, DataShapley, DataBanzhaf, \
    BetaShapley, DVRL, RobustVolumeShapley, KNNShapley, InfluenceFunction, InfluenceSubsample
from sklearn.utils import check_random_state
from torch.utils.data import Dataset


def get_dataset_split(dataset_name: str, cache_dir: str, use_original_splitting: bool = True):
    """ Returns the split for a given dataset. We mostly use the same splittings as the original
    OpenDataVal paper (=> use_original_splitting=True). Otherwise, we use an 80%-10%-10% split.

    "The sample sizes for the training and validation datasets are set to 1000 and 100, respectively. The size of the
    test dataset is fixed at 3000 for all datasets, except for the text datasets (BBC and IMDB), where it is set to 500"
    :param dataset_name: Name of dataset from OpenDataVal
    :param cache_dir: Directory where the dataset is stored (to prevent re-downloading)
    :param use_original_splitting:
    :return: Tuple of train, validation, test count
    """
    fetcher = DataFetcher(dataset_name=dataset_name, cache_dir=cache_dir, force_download=False)
    dataset_size = len(fetcher.labels)

    if use_original_splitting:
        if dataset_name in {"bbc-embeddings", "IMDB-embeddings"}:
            train_count, valid_count, test_count = 1000, 100, 500
        else:
            train_count, valid_count, test_count = 1000, 100, 3000
    else:
        train_count, valid_count, test_count = int(0.8 * dataset_size), int(0.1 * dataset_size), int(0.1 * dataset_size)

    return train_count, valid_count, test_count


def instantiate_evaluators(validation_set_size: int, nr_models: int = 1000, rl_epochs: int = 2000, k_neighbors: int = None):
    """ Instantiate the Evaluators for all baselines we use.
    :param validation_set_size: Size of the validation set (used for KNNShapley)
    :param nr_models:
    :param rl_epochs:
    :return:
    """
    # Here I want to create a new cache for every run to make sure there is no accidental overreach
    # between different runs. (To prevent that the wrong/old data valuations are re-used).
    random_number_for_cache = random.randint(0, 1000000)
    k_neighbors = validation_set_size if k_neighbors is None else k_neighbors

    # Instantiate Evaluators:
    baseline_evaluators = [
        RandomEvaluator(),
        # InfluenceFunction(),  # Only works for classification & OpenDataVal paper only uses InfluenceSubsample
        InfluenceSubsample(num_models=nr_models),
        LeaveOneOut(),
        DataShapley(cache_name=f"cached_{random_number_for_cache}"),
        BetaShapley(cache_name=f"cached_{random_number_for_cache+1}"),
        DataBanzhaf(num_models=nr_models),
        DataOob(num_models=nr_models),
        AME(num_models=nr_models),
        DVRL(rl_epochs=rl_epochs),
        KNNShapley(k_neighbors=k_neighbors),
        LavaEvaluator(),
        # RobustVolumeShapley()     # Extremely slow
    ]

    return baseline_evaluators


def create_csv_files(result_path, mode, dataset):
    """ Create csv files to store the results if they do not exist. """
    target_path = result_path + dataset + "/"

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    target_path += mode
    if not os.path.exists(target_path + "_corruption_discovery_results.csv"):
        corrupt_disc_df = pd.DataFrame(columns=["corrupt_found", "axis", "optimal", "random",
                                                "method", "dataset", "noise_rate", "model"])
        corrupt_disc_df.to_csv(target_path + "_corruption_discovery_results.csv", index=False)

    if not os.path.exists(target_path + "_f1_scores.csv"):
        f1_score_df = pd.DataFrame(columns=["kmeans_f1", "method", "dataset", "noise_rate", "model"])
        f1_score_df.to_csv(target_path + "_f1_scores.csv", index=False)

    if not os.path.exists(target_path + "_remove_high_low_results.csv"):
        rem_high_low_results_df = pd.DataFrame(columns=["remove_least_influential_first_Metrics.ACCURACY",
                                                        "remove_most_influential_first_Metrics.ACCURACY", "axis",
                                                        "method", "dataset", "noise_rate", "model"])
        rem_high_low_results_df.to_csv(target_path + "_remove_high_low_results.csv", index=False)


def mixed_noise_function(fetcher: DataFetcher, noise_rate: float = 0.2, mu: float = 0.0, sigma: float = 1.0) -> dict[str, np.ndarray]:
    """ Combine the two noise functions mix_labels and add_gauss_noise from OpenDataVal to one function.
    Make sure that the overall noise rate is respected and that only one noise function is applied per instance.
    This results in noise_rate/2 of the instances having mixed labels and noise_rate/2 having Gaussian noise.

    :param fetcher: DataFetcher object housing the data to have noise added to.
    :param noise_rate: How many instances should be noisy.
    :return: dictionary of updated data points containing the keys "y_train", "y_valid", "noisy_train_indices"
    """
    rs = check_random_state(fetcher.random_state)

    x_train = np.array(fetcher.x_train, dtype=np.float64)
    x_valid = np.array(fetcher.x_valid, dtype=np.float64)
    y_train, y_valid = fetcher.y_train, fetcher.y_valid
    num_train, num_valid = len(x_train), len(x_valid)
    feature_dim = fetcher.covar_dim

    # Select some of the instances to be noised:
    noisy_train_idx = rs.choice(num_train, round(num_train * noise_rate), replace=False)
    noisy_valid_idx = rs.choice(num_valid, round(num_valid * noise_rate), replace=False)

    # Shuffle the instances to be noised (the first half will get mixed labels, the second half Gaussian noise):
    rs.shuffle(noisy_train_idx)
    rs.shuffle(noisy_valid_idx)

    # Split the instances to be noised into two halves:
    noisy_train_idx_mix_labels = noisy_train_idx[:len(noisy_train_idx) // 2]
    noisy_train_idx_add_gauss = noisy_train_idx[len(noisy_train_idx) // 2:]
    noisy_valid_idx_mix_labels = noisy_valid_idx[:len(noisy_valid_idx) // 2]
    noisy_valid_idx_add_gauss = noisy_valid_idx[len(noisy_valid_idx) // 2:]

    # === Apply the noise functions to the instances ===
    # -- Gaussian noise --
    noise_train_gauss = rs.normal(mu, sigma, size=(len(noisy_train_idx_add_gauss), *feature_dim))
    noise_valid_gauss = rs.normal(mu, sigma, size=(len(noisy_valid_idx_add_gauss), *feature_dim))

    if isinstance(x_train, Dataset):
        # We add a zero tensor at the top because noise only some indices have noise
        # added. For those that do not, they have the zero tensor added -> no change
        padded_noise_train = np.vstack([np.zeros(shape=(1, *feature_dim)), noise_train_gauss])
        padded_noise_valid = np.vstack([np.zeros(shape=(1, *feature_dim)), noise_valid_gauss])
        noise_add_train = torch.tensor(padded_noise_train, dtype=torch.float)
        noise_add_valid = torch.tensor(padded_noise_valid, dtype=torch.float)

        # A remapping to noisy index, in noise array, offset by 1 for non-noisy data
        # as the 0th index is the zero tensor from above
        remap_train = np.zeros((num_train,), dtype=int)
        remap_valid = np.zeros((num_valid,), dtype=int)
        remap_train[noisy_train_idx_add_gauss] = range(1, len(noisy_train_idx_add_gauss) + 1)
        remap_valid[noisy_valid_idx_add_gauss] = range(1, len(noisy_valid_idx_add_gauss) + 1)

        x_train = IndexTransformDataset(x_train, lambda data, ind: (data + noise_add_train[remap_train[ind]]))
        x_valid = IndexTransformDataset(x_valid, lambda data, ind: (data + noise_add_valid[remap_valid[ind]]))

    else:
        x_train[noisy_train_idx_add_gauss] = x_train[noisy_train_idx_add_gauss] + noise_train_gauss
        x_valid[noisy_valid_idx_add_gauss] = x_valid[noisy_valid_idx_add_gauss] + noise_valid_gauss

    # -- Mixed labels --
    # Gets unique classes and mapping of training data set to those classes
    train_classes, train_mapping = np.unique(y_train, return_inverse=True, axis=0)
    valid_classes, valid_mapping = np.unique(y_valid, return_inverse=True, axis=0)

    # For each label, we determine a shift to pick a new label
    # The new label cannot be the same as the prior, therefore start at 1
    train_shift = rs.choice(len(train_classes) - 1, len(noisy_train_idx_mix_labels)) + 1
    valid_shift = rs.choice(len(valid_classes) - 1, len(noisy_valid_idx_mix_labels)) + 1

    train_noise = (train_mapping[noisy_train_idx_mix_labels] + train_shift) % len(train_classes)
    valid_noise = (valid_mapping[noisy_valid_idx_mix_labels] + valid_shift) % len(valid_classes)

    y_train[noisy_train_idx_mix_labels] = train_classes[train_noise]
    y_valid[noisy_valid_idx_mix_labels] = valid_classes[valid_noise]

    return {
        "y_train": y_train,
        "y_valid": y_valid,
        "x_train": x_train,
        "x_valid": x_valid,
        "noisy_train_indices": noisy_train_idx,
    }
