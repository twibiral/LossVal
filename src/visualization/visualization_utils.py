import itertools

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import src.visualization.create_critical_difference_diagram as cdd


def remove_cache_id(df):
    cache_re = r"cache_name=cached_\d+(?=\))"
    df['method'] = df['method'].str.replace(cache_re, "", regex=True)
    return df


def clean_up_evaluator_names(df):
    cleaner_re = r"(\s*device\s?=\s?\w+,?\s*)|(cache_name\?=\?cached_\d+,?\s*)"
    df['method'] = df['method'].str.replace(cleaner_re, "", regex=True)
    return df



def method_renaming(name_str: str, main_method_name_regr: str = "Weighted MSE loss with squared Sinkhorn", main_method_name_class: str = "Weighted CE loss with squared Sinkhorn"):
    """ Change the detailed method name to a more readable version for the plots. """
    # Extract learning rate ("lr=...") and epochs ("nr_epochs=...")
    lr = name_str.split("lr=")[1].split(")")[0]
    epochs = name_str.split("epochs=")[1].split(",")[0]

    temp_name = name_str

    name_str = name_str.replace(main_method_name_regr + ",", "LossVal,")
    name_str = name_str.replace(main_method_name_class + ",", "LossVal,")

    # Remove the evaluator part
    name_str = name_str[22:].split(",")[0]

    name_str = name_str.replace(" only", "")
    name_str = name_str.replace(" loss", "")

    # print(temp_name, "->", name_str+f" (lr={lr}, epochs={epochs})")

    return name_str + f" (lr={lr}, epochs={epochs})"


def name_mapping(df):
    """ Rename the methods to be more readable. Works on a copy of the dataframe. """
    df = df.copy()
    renaming_scheme = {
        "DataBanzhaf": "Data Banzhaf",
        "DataOob": "Data-OOB",
        "InfluenceSubsample": "Influence Subsample",
        "KNNShapley": "KNN-Shapley",
        "LeaveOneOut": "Leave-One-Out",
        "LavaEvaluator": "LAVA",
        "DataShapley": "Data Shapley",
        "BetaShapley": "Beta Shapley",
    }
    for method_name in df["method"].unique():
        if method_name in renaming_scheme:
            df['method'] = df['method'].replace(method_name, renaming_scheme[method_name])

    return df


def plot_corruption_discovery(df, noise_rate, alpha=1, n_cols=2):
    df = df.copy()
    df = remove_cache_id(df)

    # Group all columns by corrupt_found, method, dataset, noise_rate, model and take the mean
    df = df.groupby(['corrupt_found', 'method', 'dataset', 'noise_rate', 'model']).mean().reset_index()

    list_of_datasets = df['dataset'].unique()
    list_of_methods = df['method'].unique()
    print("List of datasets: ", list_of_datasets)
    print("List of methods: ", list_of_methods)

    # For each method, dataset, noise_rate add the "start point" at (0, 0) and
    # the "end point" at (1, 1) to make the plot look better
    end_points = [{'corrupt_found': 1, 'axis': 1, 'method': meth, 'dataset': ds, 'noise_rate': noise_rate} for
                  (ds, meth) in df[['dataset', 'method']].values]
    end_points_df = pd.DataFrame(end_points)

    start_points = [{'corrupt_found': 0, 'axis': 0, 'method': meth, 'dataset': ds, 'noise_rate': noise_rate} for
                    (ds, meth) in df[['dataset', 'method']].values]
    start_points_df = pd.DataFrame(start_points)

    df = pd.concat([df, start_points_df, end_points_df])

    # One subplot for each dataset; arranged in columns of 2
    n_datasets = len(list_of_datasets)
    n_cols = 2
    n_rows = n_datasets // n_cols + int(n_datasets % n_cols > 0)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 20))

    df_scaled = df.copy()
    df_scaled = df_scaled[df_scaled['noise_rate'] == noise_rate]
    df_scaled['axis'] = df_scaled['axis'] * 100
    df_scaled['corrupt_found'] = df_scaled['corrupt_found'] * 100

    for i, ds in enumerate(list_of_datasets):
        row = i // n_cols
        col = i % n_cols
        df_slice = df_scaled[df_scaled["dataset"] == ds]

        sns.lineplot(data=df_slice, x="axis", y="corrupt_found", hue="method", style="method", markers=True,
                     dashes=False, alpha=alpha, ax=axs[row, col], errorbar=None)
        axs[row, col].set_title(f"'{ds}' dataset")

        # Add "random" line and "ideal" line:
        axs[row, col].plot([0, 100], [0, 100], linestyle='--', color='black', label="random", alpha=alpha)
        axs[row, col].plot([0, noise_rate * 100, 100], [0, 100, 100], linestyle=':', color='red', label="ideal",
                           alpha=alpha)
        axs[row, col].legend()

        # Only set the labels for the outermost subplots
        if col == 0:
            axs[row, col].set_ylabel("proportion of corruption found (%)")
        else:
            axs[row, col].set_ylabel("")

        if row == n_rows - 1:
            axs[row, col].set_xlabel("proportion of data inspected (%)")
        else:
            axs[row, col].set_xlabel("")

        # Move the legend to the bottom of the plot
        handles, labels = axs[row, col].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))
        axs[row, col].get_legend().remove()  # Remove the legend from the subplot

    return fig


def plot_corruption_discovery_averaged_over_all_datasets(df, noise_rates=[0.05, 0.1, 0.15, 0.2], alpha=1, accumulation="mean"):
    """ Plots the average corruption discovery over all datasets for each method. One plot per noise rate.
    Use accumulation=median to get more 'smooth' plots.
    """
    assert accumulation in ["mean", "median"], "accumulation must be either 'mean' or 'median'."
    df = df.copy()
    df = remove_cache_id(df)

    # Group all columns by corrupt_found, method, dataset, noise_rate, model and take the mean or median
    if accumulation == "mean":
        df = df.groupby(['corrupt_found', 'method', 'noise_rate']).mean(numeric_only=True).reset_index()
    else:   # accumulation == "median"
        df = df.groupby(['corrupt_found', 'method', 'noise_rate']).median(numeric_only=True).reset_index()

    # For each method, dataset, noise_rate add the "start point" at (0, 0) and
    # the "end point" at (1, 1) to make the plot look better
    end_points = [{'corrupt_found': 1, 'axis': 1, 'method': meth, 'noise_rate': noise_rate} for
                    (noise_rate, meth) in itertools.product(noise_rates, df['method'].unique())]
    end_points_df = pd.DataFrame(end_points)

    start_points = [{'corrupt_found': 0, 'axis': 0, 'method': meth, 'noise_rate': noise_rate} for
                    (noise_rate, meth) in itertools.product(noise_rates, df['method'].unique())]
    start_points_df = pd.DataFrame(start_points)

    df = pd.concat([df, start_points_df, end_points_df])

    # One subplot for each dataset; arranged in columns of 2
    n_plots = len(noise_rates)
    n_cols = 2
    n_rows = n_plots // n_cols + int(n_plots % n_cols > 0)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))

    df_scaled = df.copy()
    df_scaled['axis'] = df_scaled['axis'] * 100
    df_scaled['corrupt_found'] = df_scaled['corrupt_found'] * 100

    for i, noise_rate in enumerate(noise_rates):
        row = i // n_cols
        col = i % n_cols
        df_slice = df_scaled[df_scaled["noise_rate"] == noise_rate]

        sns.lineplot(data=df_slice, x="axis", y="corrupt_found", hue="method", style="method", markers=True,
                     dashes=False, alpha=alpha, ax=axs[row, col], errorbar=None)
        axs[row, col].set_title(f"{100*noise_rate}% noise")

        # Add "random" line and "ideal" line:
        axs[row, col].plot([0, 100], [0, 100], linestyle='--', color='black', label="random", alpha=alpha)
        axs[row, col].plot([0, noise_rate * 100, 100], [0, 100, 100], linestyle=':', color='red', label="ideal", alpha=alpha)
        axs[row, col].legend()

        # Only set the labels for the outermost subplots
        if col == 0:
            axs[row, col].set_ylabel("proportion of corruption found (%)")
        else:
            axs[row, col].set_ylabel("")

        if row == n_rows - 1:
            axs[row, col].set_xlabel("proportion of data inspected (%)")
        else:
            axs[row, col].set_xlabel("")

        # Move the legend to the bottom of the plot
        handles, labels = axs[row, col].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))
        axs[row, col].get_legend().remove()  # Remove the legend from the subplot

    return fig


def f1_scores_visualization(df, alpha=1, n_cols=2):
    df = df.copy()
    df = remove_cache_id(df)

    list_of_datasets = df['dataset'].unique()
    list_of_methods = df['method'].unique()
    print("List of datasets: ", list_of_datasets)
    print("List of methods: ", list_of_methods)

    # Group by method, dataset, noise_rate, model and take the mean
    df = df.groupby(['method', 'dataset', 'noise_rate', 'model']).mean(numeric_only=True).reset_index()
    df_scaled = df.copy()
    df_scaled['noise_rate'] = df_scaled['noise_rate'] * 100

    n_datasets = len(list_of_datasets)

    n_rows = n_datasets // n_cols + int(n_datasets % n_cols > 0)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10 * (n_cols/2), 10 * (n_rows/2)))

    max_f1_score = df['kmeans_f1'].max()
    y_axis_max = min(1, max_f1_score+0.05)

    for i, ds in enumerate(list_of_datasets):
        row = i // n_cols
        col = i % n_cols
        df_slice = df_scaled[df_scaled["dataset"] == ds]

        sns.lineplot(data=df_slice, x="noise_rate", y="kmeans_f1", hue="method", style="method", markers=True,
                     dashes=False, alpha=alpha, ax=axs[row, col])
        axs[row, col].set_title(f"'{ds}' dataset")

        if col == 0:
            axs[row, col].set_ylabel("F1 score")
        else:
            axs[row, col].set_ylabel("")

        if row == n_rows - 1:
            axs[row, col].set_xlabel("Noise rate (%)")
        else:
            axs[row, col].set_xlabel("")

        # Set y-axis limits to [0, y_axis_max]
        axs[row, col].set_ylim([0, y_axis_max])

        # Move the legend to the bottom of the plot
        handles, labels = axs[row, col].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))
        axs[row, col].get_legend().remove()
        axs[row, col].set_xticks([5, 10, 15, 20])

    return fig


def f1_scores_averaged_visualization(df_noisy_label, df_noisy_feature, df_mixed_noise, alpha=1):
    fig, axs = plt.subplots(1, 3, figsize=(10, 10))

    df_dict = {
        "Noisy Labels": df_noisy_label,
        "Noisy Features": df_noisy_feature,
        "Mixed Noise": df_mixed_noise
    }

    max_f1_score = max(df_noisy_label["kmeans_f1"].max(),
                       df_noisy_feature["kmeans_f1"].max(),
                       df_mixed_noise["kmeans_f1"].max())
    y_axis_max = min(1, (max_f1_score + 0.05))

    for i, (name, df) in enumerate(df_dict.items()):
        df = df.copy()
        df = remove_cache_id(df)

        # Group by method, dataset, noise_rate, model and take the mean
        df = df.groupby(['method', 'noise_rate']).mean(numeric_only=True).reset_index()
        df_scaled = df.copy()
        df_scaled['noise_rate'] = df_scaled['noise_rate'] * 100

        all_noise_rates = df_scaled['noise_rate'].unique().tolist()

        sns.lineplot(data=df_scaled, x="noise_rate", y="kmeans_f1", hue="method", style="method", markers=True,
                           dashes=False, alpha=alpha)
        fig.set_title(f"F1 Scores over all datasets (regression & classification)")


        if i == 0:
            axs[0, i].set_ylabel("F1 score")
        else:
            axs[0, i].set_ylabel("")

        fig.set_xlabel("Noise rate (%)")

        # Set y-axis limits to [0, y_axis_max]
        fig.set_ylim([0, y_axis_max])

        # Move the legend to the bottom of the plot
        handles, labels = fig.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.5))
        fig.set_xticks(all_noise_rates)

    return fig



def remove_data_points(df, remove_col, noise_rate, alpha=1):
    df = df.copy()
    df = remove_cache_id(df)

    # Group by axis, method, dataset, noise_rate, model and take the mean:
    df = df.groupby(['axis', 'method', 'dataset', 'noise_rate', 'model']).mean().reset_index()

    df_sliced = df[df['noise_rate'] == noise_rate]

    n_datasets = len(df_sliced['dataset'].unique())
    n_cols = 2
    n_rows = n_datasets // n_cols + int(n_datasets % n_cols > 0)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))

    for i, ds in enumerate(df_sliced['dataset'].unique()):
        row = i // n_cols
        col = i % n_cols
        df_slice = df_sliced[df_sliced["dataset"] == ds]

        sns.lineplot(data=df_slice, x="axis", y=remove_col, hue="method", style="method", markers=True, dashes=False,
                     alpha=alpha, ax=axs[row, col])
        axs[row, col].set_title(f"'{ds}' dataset")

        if col == 0:
            axs[row, col].set_ylabel("Accuracy")
        else:
            axs[row, col].set_ylabel("")

        if row == n_rows - 1:
            axs[row, col].set_xlabel("Proportion of data removed (%)")
        else:
            axs[row, col].set_xlabel("")

        # Set the x-axis to be in the range [0, 0.8]
        axs[row, col].set_xlim([0, 0.8])
        # Set the y-axis to be in the range [0, 1]
        axs[row, col].set_ylim([0, 1])

        # Move the legend to the bottom of the plot
        handles, labels = axs[row, col].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.1))
        axs[row, col].get_legend().remove()

    return fig


def draw_and_embedd_cdd(df, target_path, title=None, labels=True):
    cdd.draw_cd_diagram(df_perf=df, title=title, labels=labels, target_path=target_path)
