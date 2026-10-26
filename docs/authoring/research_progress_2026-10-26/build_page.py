"""Build an offline public-facing summary from existing aggregate evidence only."""
import csv
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DESTINATION = ROOT / "output/briefing/research_progress_2026-10-26.html"
PILOT = ROOT / "runs/model_selection_v2/pilot_candidates_2026-10-26"
SAFE = ROOT / "runs/model_selection_v2/safe2026_swapping_2026-10-26"
INPUTS = []


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    INPUTS.append(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    INPUTS.append(path)
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name, value in row.items():
            try:
                row[name] = float(value) if value else None
            except (TypeError, ValueError):
                pass
    return rows


def main(refresh=False):
    exists = DESTINATION.exists() or DESTINATION.with_suffix(".sources.json").exists()
    if exists:
        if not refresh:
            raise FileExistsError("Choose a new version or explicitly refresh this authored HTML report.")
        previous = json.loads(DESTINATION.with_suffix(".sources.json").read_text())
        assert digest(DESTINATION) == previous["output_sha256"], "Report changed outside this authoring session."
    for folder in (PILOT, SAFE):
        marker = read_json(folder / "VERIFIED.json")
        actual = {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob("*")
                  if p.is_file() and p.name != "VERIFIED.json"}
        assert marker["status"] == "verified" and actual == marker["artifact_hashes"]
    baseline_paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).decode().split("\0")
    protected = {name: digest(ROOT / name) for name in baseline_paths if name and name not in
                 {"PROJECT_CONTEXT.md", "README.md", "AGENTS.md", DESTINATION.relative_to(ROOT).as_posix(),
                  DESTINATION.with_suffix(".sources.json").relative_to(ROOT).as_posix()} and not name.startswith("docs/authoring/research_progress_2026-10-26/")}
    pilot = read_csv(PILOT / "diagnostics/selection_metrics.csv")
    scores = read_json(SAFE / "integrated_scores.json")
    data = {"date": "26 October 2026", "pilot": {row["candidate_id"]: row for row in pilot},
            "scores": scores, "rankings": read_json(SAFE / "rankings.json"),
            "areas": {row["candidate_id"]: row for row in read_csv(SAFE / "component_areas.csv")},
            "historical": {row["model"]: row for row in read_csv(ROOT / "outputs/tables/performance_evaluation/holdout_metrics.csv")},
            "article": read_csv(ROOT / "article_safe/runs/article_v1_2026-10-10/report/model_profiles.csv"),
            "fairness": read_csv(ROOT / "outputs/tables/fairness_evaluation/fairness_gaps_threshold_050.csv"),
            "curves": {}, "paths": {}, "exposure": []}
    for candidate in data["pilot"]:
        raw = read_json(SAFE / f"curves/{candidate}.json")
        data["curves"][candidate] = {key: {"x": raw[key]["x"], "values": raw[key]["values"]}
                                     for key in ("accuracy", "explainability", "robustness")}
        path = read_json(SAFE / f"removal/{candidate}.json")
        data["paths"][candidate] = [{"step": row["step"], "removed": row["alternative"],
                                     "preservation": row["reversed_rge"]} for row in path["path"]]
    for row in read_json(SAFE / "perturbation_diagnostics.json"):
        data["exposure"].append({"raw": row["raw_strength"], "changed": row["records_with_any_change"],
            "novel": row["support"]["records_with_categorical_combination_absent_from_original_selection"]})
    for source in (ROOT / "PROJECT_CONTEXT.md", ROOT / "research/model_selection_v2/safe2026_specification.md",
                   ROOT / "outputs/fairness_evaluation_report.md", ROOT / "outputs/explainability_evaluation_report.md",
                   ROOT / "outputs/robustness_evaluation_report.md", ROOT / "article_safe/runs/article_v1_2026-10-10/report/results.md",
                   PILOT / "protocol.json", SAFE / "protocol.json", SAFE / "independent_verification.json"):
        INPUTS.append(source)
    template = (HERE / "page.template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    assert template.count("__REPORT_DATA__") == 1
    output = template.replace("__REPORT_DATA__", payload)
    with DESTINATION.open("w" if refresh else "x", encoding="utf-8", newline="\n") as stream:
        stream.write(output)
    assert all(digest(ROOT / name) == expected for name, expected in protected.items())
    manifest = {"created_date": "2026-10-26", "purpose": "Plain-language offline research results summary",
                "research_recomputed": False, "individual_record_data_embedded": False,
                "input_hashes": {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(set(INPUTS))},
                "authoring_hashes": {path.relative_to(ROOT).as_posix(): digest(path) for path in sorted(HERE.glob("*")) if path.is_file()},
                "output_sha256": digest(DESTINATION), "protected_files_checked": len(protected),
                "protected_file_hashes_unchanged": True}
    with DESTINATION.with_suffix(".sources.json").open("w" if refresh else "x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    print(DESTINATION)
    print(f"Protected files unchanged: {len(protected)}; embedded aggregate snapshot: {len(payload)} characters")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Refresh only this authored HTML after checking its existing identity")
    main(parser.parse_args().refresh)
