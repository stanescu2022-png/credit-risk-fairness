"""Evaluate frozen models; --verify repeats outputs and runs all project tests."""
import argparse
from src.explainability_evaluation import run_explainability_evaluation, verify_completion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify_completion()
    else:
        run_explainability_evaluation()
