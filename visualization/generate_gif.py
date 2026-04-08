import sys
import os
# Add the root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygame
import numpy as np
from envs.bottleneck_env import BottleneckEnv
from visualization.pygame_renderer import TrafficRenderer
from PIL import Image
import os
import sys

def generate_simulation_gif(model_path=None, output_path="simulation.gif", num_frames=120):
    """
    Runs the simulation and exports it as a high-quality GIF.
    """
    print(f"🎬 Starting GIF generation ({num_frames} frames)...")
    
    # Initialize environment and renderer
    env = BottleneckEnv(max_steps=num_frames + 50)
    renderer = TrafficRenderer(env)
    
    # Try to load the RL model
    model = None
    if model_path and os.path.exists(model_path):
        from stable_baselines3 import PPO
        try:
            model = PPO.load(model_path)
            print(f"✅ Loaded RL Model from {model_path}")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}. Falling back to rule-based.")
    else:
        print("ℹ️ No model path provided. Generating rule-based baseline GIF.")

    frames = []
    obs, _ = env.reset()

    # We skip the first few frames to let the road fill up
    print("⏳ Warming up simulation...")
    for _ in range(40):
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = None
        obs, _, _, _, _ = env.step(action)

    print("📸 Capturing frames...")
    for i in range(num_frames):
        # 1. Update simulation
        if model:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = None
        
        obs, reward, done, truncated, info = env.step(action)
        
        # 2. Render to the hidden/visible surface
        renderer.render()
        
        # 3. Capture surface to PIL Image
        # We use pygame.image.tostring and PIL.Image.frombytes for speed/compatibility
        data = pygame.image.tostring(renderer.screen, "RGB")
        img = Image.frombytes("RGB", renderer.screen.get_size(), data)
        frames.append(img)
        
        if done or truncated:
            break
        
        if i % 20 == 0:
            print(f"   Progress: {i}/{num_frames} frames")

    # Save the GIF
    print(f"💾 Saving GIF to {output_path}...")
    if frames:
        # duration=33ms is ~30fps
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=40, 
            loop=0,
            optimize=True
        )
        print(f"✨ Successfully generated: {os.path.abspath(output_path)}")
    else:
        print("❌ Error: No frames were captured.")

    pygame.quit()

if __name__ == "__main__":
    # Point to the final trained model
    MODEL_FILE = "./models/ppo_bottleneck_final.zip"
    
    # If the user wants a baseline, they can rename the file or I can add a flag
    generate_simulation_gif(
        model_path=MODEL_FILE if os.path.exists(MODEL_FILE) else None,
        output_path="simulation.gif",
        num_frames=150
    )
