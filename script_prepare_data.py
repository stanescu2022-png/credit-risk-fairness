"""Regenerate cleaned data, duplicate audit, and leakage-safe baseline split."""
import json
from src.baseline import prepare_data

if __name__ == "__main__":
    _, _, audit = prepare_data()
    print(json.dumps(audit, indent=2))
