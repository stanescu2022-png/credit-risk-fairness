"""Run the fixed Performance Evaluation protocol from any working directory."""
import matplotlib

matplotlib.use("Agg")

from src.performance_evaluation import run_performance_evaluation

if __name__ == "__main__":
    run_performance_evaluation()
