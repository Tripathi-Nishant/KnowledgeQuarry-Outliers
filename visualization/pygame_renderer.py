import pygame
import numpy as np
from envs.bottleneck_env import BottleneckEnv
from agents.behavior_profiles import PROFILES
import sys

# Constants for scaling
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 400
LANE_HEIGHT = 40
ROAD_Y = 150
WORLD_TO_PIXEL = SCREEN_WIDTH / 1000.0

# Colors
COLOR_BG = (30, 30, 30)
COLOR_ROAD = (60, 60, 60)
COLOR_LANE_MARK = (200, 200, 200)
COLOR_BOTTLENECK = (255, 50, 50)
COLOR_TEXT = (240, 240, 240)

class TrafficRenderer:
    def __init__(self, env: BottleneckEnv):
        pygame.init()
        self.env = env
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Traffic RL Bottleneck Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.paused = False

    def render(self):
        self.screen.fill(COLOR_BG)
        
        # 1. Draw Road
        road_rect = pygame.Rect(0, ROAD_Y, SCREEN_WIDTH, self.env.n_lanes * LANE_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_ROAD, road_rect)
        
        # Lane Markings
        for i in range(1, self.env.n_lanes):
            y = ROAD_Y + i * LANE_HEIGHT
            dash_len = 20
            for x in range(0, SCREEN_WIDTH, dash_len * 2):
                pygame.draw.line(self.screen, COLOR_LANE_MARK, (x, y), (x + dash_len, y), 2)
        
        # Bottleneck Marker
        bx = self.env.bottleneck_x * WORLD_TO_PIXEL
        pygame.draw.line(self.screen, COLOR_BOTTLENECK, (bx, ROAD_Y), (bx, ROAD_Y + self.env.n_lanes * LANE_HEIGHT), 3)
        
        # 2. Draw Vehicles
        for v in self.env.vehicles:
            if v.passed and v.x > self.env.road_length: continue
            
            # Position
            vx = v.x * WORLD_TO_PIXEL
            vy = ROAD_Y + v.lane * LANE_HEIGHT + (LANE_HEIGHT / 4)
            
            # Smoothing lane change
            if v.merging:
                target_y = ROAD_Y + v.target_lane * LANE_HEIGHT + (LANE_HEIGHT / 4)
                vy = vy + (target_y - vy) * v.merge_progress
            
            # Convert Hex to RGB
            color = pygame.Color(v.color)
            if v.speed < 0.2:
                # Dim when stopped
                color.hsla = (color.hsla[0], color.hsla[1], color.hsla[2] * 0.6, color.hsla[3])
                
            rect = pygame.Rect(vx - 10, vy, 20, LANE_HEIGHT / 2)
            pygame.draw.rect(self.screen, color, rect, border_radius=3)
            
            # Conflict Halo
            if v.in_conflict:
                pygame.draw.rect(self.screen, (255, 0, 0), rect.inflate(4, 4), 2, border_radius=5)

        # 3. HUD
        info = self.env._get_info()
        hud_text = [
            f"Step: {self.env.current_step} / {self.env.max_steps}",
            f"Throughput: {info['throughput']:.2f} v/s",
            f"Conflicts: {info['conflicts']}",
            f"Avg Wait: {info['avg_wait']:.1f} s",
            f"Active Vehicles: {len([v for v in self.env.vehicles if not v.passed])}",
            f"Arrival Rate: {self.env.spawner.arrival_rate:.1f} (UP/DOWN to adj)",
            f"Aggressive Ratio: {self.env.spawner.aggressive_ratio*100:.0f}% (A/D to adj)"
        ]
        
        for i, text in enumerate(hud_text):
            surface = self.font.render(text, True, COLOR_TEXT)
            self.screen.blit(surface, (20, 20 + i * 22))

        if self.paused:
            pause_surface = self.font.render("PAUSED (Space to Resume)", True, (255, 255, 100))
            self.screen.blit(pause_surface, (SCREEN_WIDTH/2 - 100, 20))

        pygame.display.flip()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused
                if event.key == pygame.K_UP:
                    self.env.spawner.arrival_rate = min(10, self.env.spawner.arrival_rate + 0.5)
                if event.key == pygame.K_DOWN:
                    self.env.spawner.arrival_rate = max(0.5, self.env.spawner.arrival_rate - 0.5)
                if event.key == pygame.K_a:
                    self.env.spawner.aggressive_ratio = max(0, self.env.spawner.aggressive_ratio - 0.05)
                if event.key == pygame.K_d:
                    self.env.spawner.aggressive_ratio = min(1, self.env.spawner.aggressive_ratio + 0.05)
                if event.key == pygame.K_r:
                    self.env.reset()

def run_visual_demo(model_path=None):
    from stable_baselines3 import PPO
    env = BottleneckEnv(max_steps=10000) # Longer for demo
    renderer = TrafficRenderer(env)
    
    model = None
    if model_path and os.path.exists(model_path):
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")

    obs, _ = env.reset()
    while True:
        renderer.handle_events()
        
        if not renderer.paused:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = None # Use rule-based purely
            
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                obs, _ = env.reset()
        
        renderer.render()
        renderer.clock.tick(30) # 30 FPS

if __name__ == "__main__":
    import os
    # Try to load model if exists, else run purely rule-based
    model_file = "./models/ppo_bottleneck_final.zip"
    run_visual_demo(model_file if os.path.exists(model_file) else None)
