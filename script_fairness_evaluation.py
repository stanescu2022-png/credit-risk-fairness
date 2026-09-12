"""Reproduce the frozen-model Fairness Evaluation from any working directory."""
import matplotlib
matplotlib.use("Agg")
from src.fairness_evaluation import run_fairness_evaluation

if __name__ == "__main__":
    run_fairness_evaluation()
