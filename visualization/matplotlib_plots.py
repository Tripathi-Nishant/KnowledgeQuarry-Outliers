import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

sns.set_theme(style="whitegrid")

def plot_tipping_point(csv_path="./experiments/results/tipping_point.csv"):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x="aggressive_ratio", y="throughput", marker="o", lw=2)
    
    # Highlight collapse zone (roughly 0.3 to 0.4)
    plt.axvspan(0.25, 0.40, color='red', alpha=0.1, label="Tipping Zone")
    plt.title("Traffic Throughput vs Aggression Ratio", fontsize=14)
    plt.xlabel("Aggressive Driver Ratio", fontsize=12)
    plt.ylabel("Throughput (v/s)", fontsize=12)
    plt.ylim(0, max(df["throughput"]) * 1.2)
    plt.legend()
    plt.savefig("./experiments/results/tipping_point.png", dpi=300)
    plt.close()

def plot_experiment_comparison(csv_path="./experiments/results/comparison.csv"):
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    
    # 4 Subplots for 4 metrics
    metrics = ["throughput", "avg_wait", "conflicts", "avg_speed"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = sns.color_palette("husl", len(df))
    
    for i, metric in enumerate(metrics):
        sns.barplot(data=df, x="experiment", y=metric, ax=axes[i], palette=colors)
        axes[i].set_title(f"Comparison: {metric.replace('_', ' ').title()}", fontsize=12)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].set_xlabel("")
        
    plt.tight_layout()
    plt.savefig("./experiments/results/experiment_comparison.png", dpi=300)
    plt.close()

def plot_speed_distribution(episode_data_list):
    """
    KDE lot of speeds for different scenarios.
    Expects a list of dictionaries with labels and speed arrays.
    """
    plt.figure(figsize=(10, 6))
    for entry in episode_data_list:
        sns.kdeplot(entry["speeds"], label=entry["label"], fill=True, alpha=0.3)
    
    plt.title("Vehicle Speed Distributions", fontsize=14)
    plt.xlabel("Speed (units/step)")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig("./experiments/results/speed_distribution.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # Examples of running if results exist
    plot_tipping_point()
    plot_experiment_comparison()
