"""Evaluate frozen models; --verify runs twice and checks all artifacts."""
import argparse
from src.robustness_evaluation import run_robustness_evaluation, verify_completion

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify_completion() if args.verify else run_robustness_evaluation()
