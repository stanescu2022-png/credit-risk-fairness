"""Labelled scientific projections; the full six-vector determines dominance."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

from .protocol import CANDIDATES

LABELS = {"logistic": "LR", "decision_tree": "DT", "random_forest": "RF", "gradient_boosting": "GBM"}
COLORS = {"logistic": "#0072B2", "decision_tree": "#E69F00", "random_forest": "#009E73", "gradient_boosting": "#CC79A7"}


def make_plots(folder, point, q):
    folder = Path(folder)
    coordinates = point["coordinates"]
    principal = point["comparisons"]["principal_contrast"]
    for filename, xname, yname, xlabel, ylabel in (
        ("probability_projection.png", "clean_brier", "worst_joint_brier", "Clean Brier", "Worst joint mean Brier"),
        ("disparity_projection.png", "absolute_tpr_gap", "absolute_fpr_gap", "Absolute clean TPR sex gap", "Absolute clean FPR sex gap"),
    ):
        fig, ax = plt.subplots(figsize=(9, 6.5))
        available = [(coordinates[candidate][xname], coordinates[candidate][yname]) for candidate in CANDIDATES
                     if coordinates[candidate][xname] is not None and coordinates[candidate][yname] is not None]
        xmid = (min(x for x, _ in available)+max(x for x, _ in available))/2 if available else 0
        ymid = (min(y for _, y in available)+max(y for _, y in available))/2 if available else 0
        annotations = {}
        for candidate in CANDIDATES:
            x, y = coordinates[candidate][xname], coordinates[candidate][yname]
            if x is None or y is None:
                continue
            ax.scatter(x, y, s=65, marker="o", color=COLORS[candidate], zorder=3)
            if candidate == principal["SAFE_choice_id"]:
                ax.scatter(x, y, s=210, marker="s", facecolors="none", edgecolors="black", linewidths=1.4, zorder=4)
            if candidate == principal["our_choice_id"]:
                ax.scatter(x, y, s=105, marker="D", facecolors="none", edgecolors="black", linewidths=1.4, zorder=5)
            annotations.setdefault((x, y), []).append(LABELS[candidate])
        ax.set(xlabel=xlabel+" (lower is better)", ylabel=ylabel+" (lower is better)",
               title=f"q={q:.2f} {'reference' if q == .10 else 'sensitivity'}: two-coordinate projection")
        ax.grid(alpha=.2)
        ax.margins(x=.25, y=.25)
        fig.subplots_adjust(left=.12, right=.96, top=.88, bottom=.29)
        fig.text(.5, .205, "Projection only: consult the full six-coordinate Pareto results. Choices stay fixed at q=.10 identities.", ha="center", fontsize=9)
        fig.legend(handles=[Line2D([], [], color=COLORS[candidate], marker="o", linestyle="none", label=f"{LABELS[candidate]}: {candidate}")
                            for candidate in CANDIDATES], loc="lower center", bbox_to_anchor=(.5, .10), ncol=4, frameon=False, fontsize=8)
        fig.legend(handles=[Line2D([], [], color="black", marker="s", markerfacecolor="none", linestyle="none", label="Frozen arithmetic-SAFE choice"),
                            Line2D([], [], color="black", marker="D", markerfacecolor="none", linestyle="none", label="Frozen full-screen choice")],
                   loc="lower center", bbox_to_anchor=(.5, .04), ncol=2, frameon=False, fontsize=9)
        # Only text positions move. Choose deterministic offsets that avoid
        # marker/text collisions and keep short labels inside the axes.
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        occupied = []
        marker_boxes = [Bbox.from_bounds(px-10, py-10, 20, 20) for px, py in (ax.transData.transform(xy) for xy in available)]
        for (x, y), labels in annotations.items():
            dx, dy = (-12 if x >= xmid else 12), (-14 if y >= ymid else 14)
            candidates = [(dx, dy), (dx, -dy), (-dx, dy), (-dx, -dy),
                          (dx, 30), (dx, -30), (-dx, 30), (-dx, -30), (dx, 46), (dx, -46)]
            for position, (horizontal, vertical) in enumerate(candidates):
                text = ax.annotate(" / ".join(labels), (x, y), xytext=(horizontal, vertical), textcoords="offset points",
                                   ha="right" if horizontal < 0 else "left", va="top" if vertical < 0 else "bottom", fontsize=10)
                box = text.get_window_extent(renderer).expanded(1.05, 1.1)
                inside = ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
                clear = not any(box.overlaps(other) for other in (*occupied, *marker_boxes))
                if (inside and clear) or position == len(candidates)-1:
                    occupied.append(box)
                    break
                text.remove()
        fig.savefig(folder / filename, dpi=150)
        plt.close(fig)
