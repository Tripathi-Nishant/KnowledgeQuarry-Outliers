import sys
import os
# Add root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from envs.bottleneck_env import BottleneckEnv
from training.curriculum import CurriculumManager

def train_ppo(hyperparams_path="training/hyperparams.yaml"):
    # Load configuration
    with open(hyperparams_path, 'r') as f:
        config = yaml.safe_load(f)
    
    curriculum = CurriculumManager(hyperparams_path)
    
    # Setup directories
    os.makedirs("./models/best", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    
    # Initial stage set-up
    env_kwargs = curriculum.get_stage_env_kwargs()
    # Using 4 environments for CPU parallelization
    env = make_vec_env(lambda: BottleneckEnv(**env_kwargs), n_envs=4)
    
    # Model Setup
    ppo_cfg = config['ppo']
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=ppo_cfg['learning_rate'],
        n_steps=ppo_cfg['n_steps'],
        batch_size=ppo_cfg['batch_size'],
        n_epochs=ppo_cfg['n_epochs'],
        gamma=ppo_cfg['gamma'],
        gae_lambda=ppo_cfg['gae_lambda'],
        clip_range=ppo_cfg['clip_range'],
        ent_coef=ppo_cfg['ent_coef'],
        vf_coef=ppo_cfg['vf_coef'],
        max_grad_norm=ppo_cfg['max_grad_norm'],
        policy_kwargs=dict(
            net_arch=[dict(pi=ppo_cfg['net_arch']['pi'], vf=ppo_cfg['net_arch']['vf'])]
        ),
        verbose=1,
        tensorboard_log="./logs/ppo_bottleneck",
        device="cpu" # Explicitly use CPU as requested
    )
    
    # Calibration / Sanity Check Run
    print(f"Starting Curriculum: {curriculum.get_current_stage()['name']}")
    
    # Evaluation Environment (Single thread for accuracy)
    eval_env = BottleneckEnv(**env_kwargs)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Curriculum Loop
    for i in range(len(curriculum.stages)):
        stage = curriculum.get_current_stage()
        print(f"\n--- Training Stage {i+1}: {stage['name']} ---")
        
        # Update environment if it's not the first stage
        if i > 0:
            env_kwargs = curriculum.get_stage_env_kwargs()
            print(f"Updating Environment: Aggression Ratio={env_kwargs['aggressive_ratio']}, Arrival Rate={env_kwargs['arrival_rate']}")
            env = make_vec_env(lambda: BottleneckEnv(**env_kwargs), n_envs=4)
            model.set_env(env)
            
        # Learn for stage timesteps
        model.learn(
            total_timesteps=stage['timesteps'],
            callback=eval_callback,
            reset_num_timesteps=False, # Keep cumulative timesteps
            progress_bar=True
        )
        
        model.save(f"./models/ppo_bottleneck_stage_{i+1}")
        curriculum.next_stage()

    model.save("./models/ppo_bottleneck_final")
    print("Training Complete!")

if __name__ == "__main__":
    train_ppo()
