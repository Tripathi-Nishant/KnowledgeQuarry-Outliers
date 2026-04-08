import sys
import os
# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import gymnasium as gym
from stable_baselines3 import PPO
from envs.bottleneck_env import BottleneckEnv
from experiments.metrics import summarize_metrics
import os
import tqdm

def run_episode(config, model=None, n_episodes=5):
    """
    Run simulation episodes with given config.
    Config contains environment parameters.
    If model is None, it uses rule-based defaults from environment. 
    (The environment itself handles rule-based background vehicles; 
    we need to handle the 'rl' vehicles in simulation).
    """
    # Copy cfg and remove 'name' if exists as it's not a kwarg for BottleneckEnv
    env_cfg = config.copy()
    env_name = env_cfg.pop('name', 'simulation-episode')
    env = BottleneckEnv(**env_cfg)
    results = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                # Without model, we just use random action or Maintain (1)
                action = 1 # Maintain speed
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
        results.append(summarize_metrics(info))
        
    env.close()
    return pd.DataFrame(results).mean().to_dict()

def execute_experiments():
    os.makedirs("./experiments/results", exist_ok=True)
    
    # 1. & 2. & 3. Baselines (No RL)
    print("Running Baselines (No RL)...")
    baselines = [
        {"name": "Baseline-Cooperative", "aggressive_ratio": 0.0, "cooperative_ratio": 1.0, "rl_ratio": 0.0},
        {"name": "Baseline-Aggressive", "aggressive_ratio": 1.0, "cooperative_ratio": 0.0, "rl_ratio": 0.0},
        {"name": "Baseline-Mixed", "aggressive_ratio": 0.3, "cooperative_ratio": 0.7, "rl_ratio": 0.0},
    ]
    
    baseline_results = []
    for cfg in baselines:
        print(f"Executing: {cfg['name']}")
        res = run_episode(cfg, model=None, n_episodes=20)
        res['experiment'] = cfg['name']
        baseline_results.append(res)
    
    pd.DataFrame(baseline_results).to_csv("./experiments/results/baselines.csv", index=False)

    # 4. Tipping Point Study (Aggression Sweep)
    print("\nRunning Tipping Point Study...")
    sweep_results = []
    for agg_ratio in tqdm.tqdm(np.arange(0, 1.05, 0.10)):
        cfg = {"aggressive_ratio": agg_ratio, "cooperative_ratio": 1.0 - agg_ratio, "rl_ratio": 0.0}
        res = run_episode(cfg, model=None, n_episodes=15)
        res['aggressive_ratio'] = agg_ratio
        sweep_results.append(res)
    
    pd.DataFrame(sweep_results).to_csv("./experiments/results/tipping_point.csv", index=False)

    # 5. RL Intervention Study
    # Check if model exists
    model_path = "./models/ppo_bottleneck_final.zip"
    if os.path.exists(model_path):
        print("\nRunning RL Intervention Study...")
        model = PPO.load(model_path)
        rl_cfg = {"aggressive_ratio": 0.3, "cooperative_ratio": 0.5, "rl_ratio": 0.2, "name": "RL-Intervention"}
        res = run_episode(rl_cfg, model=model, n_episodes=50)
        res['experiment'] = "RL-Intervention (20%)"
        
        # Save to comparison report
        df_base = pd.read_csv("./experiments/results/baselines.csv")
        df_rl = pd.DataFrame([res])
        pd.concat([df_base, df_rl]).to_csv("./experiments/results/comparison.csv", index=False)
    else:
        print("\nSkipping RL Experiment: Model not found yet.")

if __name__ == "__main__":
    execute_experiments()
