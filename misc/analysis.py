# -*- coding: utf-8 -*-
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Ensure seaborn is set up
sns.set_theme(style="darkgrid")


def find_json_files(root_dir):
    json_files = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith("rainbow_env.json"):
                json_files.append(os.path.join(subdir, file))
    return json_files


def generate_aggregated_rewards_plot(rewards_df):
    # Aggregate rewards by Run and Bin, summing across all locations
    aggregated_rewards = (
        rewards_df.groupby(["Run", "Bin"]).agg({"Reward": "sum"}).reset_index()
    )

    # Create a boxplot for the sum of rewards per bin
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=aggregated_rewards, x="Bin", y="Reward", showfliers=False)
    plt.title("Sum of Rewards Across All Locations per 100 Runs")
    plt.savefig(os.path.join("analysis", "sum_rewards_per_100_runs.png"))
    plt.close()


def extract_epsilon_data(root_dir):
    epsilon_values_over_time = []
    epsilons_by_location = {}

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == "training_log.json":
                file_path = os.path.join(subdir, file)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    # Assuming data is a dictionary that contains the lists directly
                    if "epsilon_values" in data:
                        epsilon_values_over_time.extend(data["epsilon_values"])
                    if "epsilons_by_location" in data:
                        for location, epsilon in data["epsilons_by_location"].items():
                            if location not in epsilons_by_location:
                                epsilons_by_location[location] = []
                            epsilons_by_location[location].append(epsilon)

    return epsilon_values_over_time, epsilons_by_location


def plot_epsilon_values(epsilon_values_over_time):
    plt.figure(figsize=(12, 8))
    plt.plot(epsilon_values_over_time, label="Epsilon Values Over Time")
    plt.xlabel("Time")
    plt.ylabel("Epsilon Value")
    plt.title("Epsilon Values Over Time")
    plt.legend()
    plt.savefig(os.path.join("analysis", "epsilon_values_over_time.png"))
    plt.close()


def plot_epsilons_by_location(epsilons_by_location):
    locations = list(epsilons_by_location.keys())
    means = [np.mean(epsilons_by_location[loc]) for loc in locations]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=locations, y=means)
    plt.xlabel("Location")
    plt.ylabel("Average Epsilon Value")
    plt.title("Average Epsilon Values by Location")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join("analysis", "epsilons_by_location.png"))
    plt.close()


def extract_data(json_files):
    all_runs = []
    rewards_data = []
    button_presses = []
    cumulative_entries = (
        0  # To keep track of the total number of entries processed so far
    )

    for file_path in json_files:
        with open(file_path, "r") as file:
            data = json.load(file)
            file_entries = (
                len(data.keys()) - 1
            )  # Subtracting one to ignore entry "1" (and "0" if present)
            for key in data:
                if key not in ["1", "0"]:  # Ignore entries "1" and "0"
                    # Adjust run_id to include cumulative entries from previous files
                    run_id = cumulative_entries + int(key)
                    entry = data[key]
                    for location, rewards in entry["rewards_per_location"].items():
                        for time_step, reward in enumerate(rewards):
                            rewards_data.append(
                                {
                                    "Run": run_id,
                                    "Location": location,
                                    "Bin": run_id // 1000,  # Bin by every 100 runs
                                    "Reward": reward,
                                }
                            )
                    all_runs.append(
                        {
                            "Run": run_id,
                            "Bin": run_id // 1000,  # Bin by every 100 runs
                            "Visited_XY": entry["visited_xy"],
                            "Timeout": entry["timeout"],
                            "Run_Time": entry["run_time"],
                        }
                    )
                    for button in entry["buttons"]:
                        button_presses.append(
                            {
                                "Run": run_id,
                                "Button": button,
                                "Bin": run_id // 1000,  # Bin by every 100 runs
                            }
                        )
            cumulative_entries += (
                file_entries  # Update cumulative entries for next file
            )

    runs_df = pd.DataFrame(all_runs)
    rewards_df = pd.DataFrame(rewards_data)
    buttons_df = pd.DataFrame(button_presses)
    return runs_df, rewards_df, buttons_df


def generate_plots(runs_df, rewards_df, buttons_df):
    output_dir = "analysis"
    os.makedirs(output_dir, exist_ok=True)

    # Metrics over time (Visited XY, Timeout, Run Time) without outliers
    metrics = ["Visited_XY", "Timeout", "Run_Time"]
    for metric in metrics:
        plt.figure(figsize=(12, 8))  # Adjusted for better visibility
        sns.boxplot(
            data=runs_df, x="Bin", y=metric, showfliers=False
        )  # Outliers not shown
        plt.title(f"{metric} per 100 Runs")
        plt.savefig(os.path.join(output_dir, f"{metric}_per_100_runs.png"))
        plt.close()

    # Rewards per location without outliers
    plt.figure(figsize=(12, 8))  # Adjusted for better visibility
    sns.boxplot(
        data=rewards_df, x="Bin", y="Reward", hue="Location", showfliers=False
    )  # Outliers not shown
    plt.title("Rewards per Location per 100 Runs")
    plt.savefig(os.path.join(output_dir, "rewards_per_location_per_100_runs.png"))
    plt.close()

    # Button presses facet plot, adjusted for wider plots and angled x-axis titles
    button_counts = (
        buttons_df.groupby(["Bin", "Button"]).size().reset_index(name="Count")
    )
    g = sns.FacetGrid(
        button_counts, col="Bin", col_wrap=4, height=4, sharex=True, aspect=2
    )  # Adjusted aspect for wider plots
    g.map_dataframe(
        lambda data, **kws: sns.barplot(
            data=data, x="Button", y="Count", **kws
        ).set_xticklabels(data["Button"], rotation=45)
    )
    g.set_titles(col_template="Bin: {col_name}")
    g.set_axis_labels("Button", "Frequency")
    plt.subplots_adjust(bottom=0.2)  # Adjust spacing to prevent clipping of tick-labels
    plt.savefig(os.path.join(output_dir, "button_presses_per_100_runs.png"))
    plt.close()

    generate_aggregated_rewards_plot(rewards_df)

    epsilon_values_over_time, epsilons_by_location = extract_epsilon_data(root_dir)
    plot_epsilon_values(epsilon_values_over_time)
    plot_epsilons_by_location(epsilons_by_location)


# Main function to orchestrate the steps
def main(root_dir):
    json_files = find_json_files(root_dir)
    # Corrected to unpack all three returned DataFrames
    runs_df, rewards_df, buttons_df = extract_data(json_files)
    generate_plots(runs_df, rewards_df, buttons_df)

    # New epsilon data extraction and plotting


if __name__ == "__main__":
    root_dir = "./"  # Ensure this points to the correct directory where your JSON files are located
    main(root_dir)
