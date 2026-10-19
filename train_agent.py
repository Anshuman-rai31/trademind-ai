"""TradeMind Root RL Agent Training Script.

Convenience launcher for Phase 6 PPO Reinforcement Learning Training & Evaluation.

Examples:
    python train_agent.py --smoke-test
    python train_agent.py --symbol TCS.NS --timesteps 5000 --seed 42
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app.rl.cli import main

if __name__ == "__main__":
    sys.exit(main())
