import yaml
import os

class CurriculumManager:
    def __init__(self, hyperparams_path):
        with open(hyperparams_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.stages = self.config['curriculum']
        self.current_stage_idx = 0

    def get_current_stage(self):
        return self.stages[self.current_stage_idx]

    def next_stage(self):
        if self.current_stage_idx < len(self.stages) - 1:
            self.current_stage_idx += 1
            return True
        return False

    def get_stage_env_kwargs(self):
        stage = self.get_current_stage()
        # Merge global env config with stage specific overrides
        env_kwargs = self.config['env'].copy()
        env_kwargs['aggressive_ratio'] = stage['aggressive_ratio']
        env_kwargs['arrival_rate'] = stage['arrival_rate']
        return env_kwargs
