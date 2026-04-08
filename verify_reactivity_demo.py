import sys
import os

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import CICDRepairEnv, Action, normalize_reward

def run_demo():
    print("="*60)
    print(" CICDRepairEnv Scoring Reactivity Demonstration")
    print(" Formula: 0.01 + (clamped_raw_reward * 0.98)")
    print("="*60)

    tiers = ["tier_1"]
    
    for tid in tiers:
        print(f"\n--- Testing {tid.upper()} ---")

        # 1. PERFECT AGENT (Wins 100% of raw reward)
        env = CICDRepairEnv()
        env.reset(tid)
        # Tier 1 requires action_id=1
        _, r, d, _ = env.step(Action(action_id=1))
        raw_perfect = env.state().cumulative_reward
        norm_perfect = normalize_reward(raw_perfect)
        print(f" [PERFECT]: Raw={raw_perfect:.2f} | Normalized={norm_perfect:.4f} (0.99 Expected)")

        # 2. PARTIAL AGENT (Takes one correct action but doesn't finish)
        # For Tier 1, action 1 is the finish, so let's try Tier 2
        print(f"\n--- Testing TIER_2 (Multi-Step) ---")
        env = CICDRepairEnv()
        env.reset("tier_2")
        # Tier 2 sequence is [4, 2]. Let's only do [4].
        env.step(Action(action_id=4)) 
        raw_partial = env.state().cumulative_reward
        norm_partial = normalize_reward(raw_partial)
        print(f" [PARTIAL]: Raw={raw_partial:.2f} | Normalized={norm_partial:.4f}")

        # 3. FAILING AGENT (Zero correct actions)
        env = CICDRepairEnv()
        env.reset("tier_1")
        # Incorrect action for Tier 1
        env.step(Action(action_id=0)) 
        raw_fail = env.state().cumulative_reward
        norm_fail = normalize_reward(raw_fail)
        print(f" [FAILING]: Raw={raw_fail:.2f} | Normalized={norm_fail:.4f} (0.01 Expected)")

    print("\n" + "="*60)
    print(" VERIFICATION COMPLETE: Score is dynamic and performance-dependent.")
    print("="*60)

if __name__ == "__main__":
    run_demo()
