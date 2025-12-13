"""
CIGRL Evaluation Script

Evaluate trained policies on various missions.
"""

import argparse
import numpy as np
from pathlib import Path


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate CIGRL policy")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained policy")
    parser.add_argument("--num-drones", type=int, default=5,
                        help="Number of drones in swarm")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--mission", type=str, default="surveillance",
                        choices=["surveillance", "delivery", "formation", "canyon"],
                        help="Mission type")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()
    
    print(f"CIGRL Evaluation")
    print(f"================")
    print(f"Model: {args.model}")
    print(f"Mission: {args.mission}")
    print(f"Drones: {args.num_drones}")
    
    np.random.seed(args.seed)
    
    from cigrl import UrbanEnvironment, DroneSwarm, CIGRLPolicy
    
    # Load policy
    policy = CIGRLPolicy.load(args.model)
    print(f"Loaded policy: {policy}")
    
    # Create environment based on mission
    if args.mission == "canyon":
        gps_zones = [(500, y, 80) for y in range(100, 900, 100)]
    else:
        gps_zones = None
        
    env = UrbanEnvironment(size=1000, n_buildings=30, 
                           gps_denied_zones=gps_zones, seed=args.seed)
    swarm = DroneSwarm(n_drones=args.num_drones, env=env)
    
    # Evaluation loop
    print("\nEvaluating...")
    results = []
    
    for episode in range(args.episodes):
        obs = swarm.reset()
        episode_reward = 0
        
        for step in range(1000):
            actions = policy.select_actions(obs, deterministic=True)
            obs, rewards, dones, info = swarm.step(actions)
            episode_reward += np.sum(rewards)
            
            if all(dones):
                break
        
        results.append({
            "reward": episode_reward,
            "steps": step + 1,
            "completion_rate": info.get("completion_rate", 0)
        })
        print(f"  Episode {episode + 1}: reward={episode_reward:.2f}, steps={step + 1}")
    
    # Summary
    print("\nResults Summary")
    print("-" * 40)
    print(f"Mean reward: {np.mean([r['reward'] for r in results]):.2f}")
    print(f"Mean steps: {np.mean([r['steps'] for r in results]):.1f}")
    print(f"Mean completion: {np.mean([r['completion_rate'] for r in results]):.1%}")


if __name__ == "__main__":
    main()
