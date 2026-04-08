import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from envs.bottleneck_env import BottleneckEnv

def main():
    print("Verifying BottleneckEnv...")
    env = BottleneckEnv()
    try:
        check_env(env)
        print("Environment verification SUCCESSFUL!")
    except Exception as e:
        print(f"Environment verification FAILED: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    main()
