import numpy as np

def calculate_throughput(passed_count, total_steps, dt=0.1):
    """Vehicles per second."""
    if total_steps == 0: return 0.0
    return passed_count / (total_steps * dt)

def calculate_gini(speeds):
    """Gini coefficient of speeds (0=equal, 1=unequal)."""
    if len(speeds) == 0: return 0.0
    # Standard Gini formula
    sorted_speeds = np.sort(speeds)
    n = len(speeds)
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * sorted_speeds)) / (n * np.sum(sorted_speeds))

def calculate_conflict_rate(conflicts, total_vehicles, total_steps):
    """Conflicts per vehicle per 100 steps."""
    if total_vehicles == 0 or total_steps == 0: return 0.0
    return (conflicts / total_vehicles) * (100 / total_steps)

def summarize_metrics(episode_data):
    """
    Summarize a collection of metrics from an episode run.
    """
    throughput = episode_data.get("total_passed", 0) / (episode_data.get("current_step", 1) * 0.1)
    avg_speed = episode_data.get("avg_speed", 0.0)
    conflicts = episode_data.get("conflicts", 0)
    avg_wait = episode_data.get("avg_wait", 0.0)
    
    return {
        "throughput": throughput,
        "avg_speed": avg_speed,
        "conflicts": conflicts,
        "avg_wait": avg_wait
    }
