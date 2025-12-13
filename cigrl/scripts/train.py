"""
CIGRL Training Script

Train multi-drone navigation policies using MAPPO.
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(description="Train CIGRL policy")
    parser.add_argument("--config", type=str, default="configs/training.yaml",
                        help="Path to training configuration")
    parser.add_argument("--num-drones", type=int, default=5,
                        help="Number of drones in swarm")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of training episodes")
    parser.add_argument("--save-path", type=str, default="models/trained_policy.npz",
                        help="Path to save trained policy")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    print(f"CIGRL Training")
    print(f"==============")
    print(f"Config: {args.config}")
    print(f"Drones: {args.num_drones}")
    print(f"Episodes: {args.episodes}")
    
    np.random.seed(args.seed)
    
    # Import here to avoid circular imports
    from cigrl import UrbanEnvironment, DroneSwarm, CIGRLPolicy
    
    # Create environment and swarm
    env = UrbanEnvironment(size=1000, n_buildings=30, seed=args.seed)
    swarm = DroneSwarm(n_drones=args.num_drones, env=env)
    policy = CIGRLPolicy()
    
    # Training loop
    print("\nTraining...")
    all_rewards = []
    
    for episode in range(args.episodes):
        obs = swarm.reset()
        episode_reward = 0
        
        for step in range(1000):
            actions = policy.select_actions(obs, deterministic=False)
            obs, rewards, dones, info = swarm.step(actions)
            episode_reward += np.sum(rewards)
            
            if all(dones):
                break
        
        all_rewards.append(episode_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(all_rewards[-100:])
            print(f"  Episode {episode + 1}: avg_reward = {avg_reward:.2f}")
    
    # Save policy
    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(save_path))
    print(f"\nSaved policy to {save_path}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
