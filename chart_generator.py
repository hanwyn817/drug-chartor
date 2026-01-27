"""
Chart generator module for creating static stability trend charts
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import matplotlib.pyplot as plt
import numpy as np

from config import (
    CHART_WIDTH,
    CHART_TITLE_FONT_SIZE,
    CHART_AXIS_FONT_SIZE,
    CHART_LEGEND_FONT_SIZE,
    CHARTS_DIR,
)


class ChartGenerator:
    """Generate static PNG charts for stability data"""

    def __init__(self):
        self.charts_generated = []

    @staticmethod
    def _set_style() -> None:
        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "font.size": CHART_AXIS_FONT_SIZE,
                "axes.titlesize": CHART_AXIS_FONT_SIZE + 2,
                "axes.labelsize": CHART_AXIS_FONT_SIZE,
                "legend.fontsize": CHART_LEGEND_FONT_SIZE,
                "axes.grid": True,
                "grid.color": "#e5e7eb",
                "grid.linestyle": "-",
                "grid.linewidth": 0.8,
                "axes.edgecolor": "#d1d5db",
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.facecolor": "white",
                "figure.facecolor": "white",
                "lines.linewidth": 2.0,
                "lines.markersize": 5.0,
                # Font fallback for CJK
                "font.sans-serif": [
                    "PingFang SC",
                    "Noto Sans CJK SC",
                    "Microsoft YaHei",
                    "SimHei",
                    "DejaVu Sans",
                ],
            }
        )

    @staticmethod
    def _parse_month(tp: str) -> Optional[int]:
        match = re.search(r"\d+", str(tp))
        if match:
            return int(match.group())
        return None

    def _build_time_axis(self, data_list: List[Dict]) -> List[str]:
        seen = {}
        for data in data_list:
            for tp in data.get("time_points", []):
                label = str(tp)
                if label not in seen:
                    seen[label] = self._parse_month(label)
        items = list(seen.items())
        items.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else 0, x[0]))
        return [label for label, _ in items]

    @staticmethod
    def _get_palette() -> List[str]:
        return [
            "#2563eb", "#16a34a", "#f59e0b", "#ef4444", "#8b5cf6",
            "#0ea5e9", "#f97316", "#14b8a6", "#db2777", "#64748b",
        ]

    @staticmethod
    def _figure_size(n_rows: int) -> Tuple[float, float]:
        width_in = max(10.0, CHART_WIDTH / 100.0)
        height_in = max(3.0, n_rows * 3.0)
        return width_in, height_in

    @staticmethod
    def _apply_limits(ax, lower: Optional[float], upper: Optional[float]) -> None:
        if lower is None and upper is None:
            return
        if lower is not None:
            ax.axhline(lower, color="#9ca3af", linestyle="--", linewidth=1.2, label="下限")
        if upper is not None:
            ax.axhline(upper, color="#9ca3af", linestyle="--", linewidth=1.2, label="上限")
        if lower is not None and upper is not None and lower < upper:
            x_min, x_max = ax.get_xlim()
            ax.fill_between(
                [x_min, x_max],
                lower,
                upper,
                color="#e5e7eb",
                alpha=0.35,
                zorder=0,
            )

    def create_single_batch_chart(
        self,
        data: Dict,
        save: bool = True,
        output_dir: Optional[Path] = None
    ) -> Path:
        """Create a chart for a single batch and save as PNG"""
        if output_dir is None:
            output_dir = CHARTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        self._set_style()

        product = data["product_name"]
        batch = data["batch_number"]
        market = data["market_standard"]
        condition = data["test_condition"]
        time_points = data["time_points"]
        test_items = data["test_items"]
        test_limits = data.get("test_limits", {})

        n_items = len(test_items)
        fig, axes = plt.subplots(n_items, 1, figsize=self._figure_size(n_items), sharex=True)
        if n_items == 1:
            axes = [axes]

        x = np.arange(len(time_points))
        for idx, (item_name, values) in enumerate(test_items.items()):
            ax = axes[idx]
            y = np.array([v if v is not None else np.nan for v in values], dtype=float)
            ax.plot(x, y, marker="o", color="#2563eb")
            ax.set_title(item_name, loc="left", fontweight="bold")
            ax.set_ylabel(item_name)

            limits = test_limits.get(item_name, {})
            lower = limits.get("lower")
            upper = limits.get("upper")
            self._apply_limits(ax, lower, upper)

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(time_points, rotation=0)
        axes[-1].set_xlabel("时间点")

        title = f"{product} - {batch}\n{market} | {condition}"
        temp_humidity = data.get("temperature_humidity", "未说明")
        if temp_humidity != "未说明":
            title += f"\n{temp_humidity}"
        fig.suptitle(title, fontsize=CHART_TITLE_FONT_SIZE, fontweight="bold", y=0.98)

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        filename = f"{product}_{batch}.png"
        filepath = output_dir / filename
        if save:
            fig.savefig(filepath, bbox_inches="tight")
            self.charts_generated.append(filepath)
            print(f"  ✓ Chart saved: {filepath.name}")
        plt.close(fig)
        return filepath

    def create_combined_chart(
        self,
        data_list: List[Dict],
        save: bool = True,
        output_dir: Optional[Path] = None
    ) -> Path:
        """Create a combined chart for multiple batches with same grouping"""
        if output_dir is None:
            output_dir = CHARTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        if not data_list:
            raise ValueError("No data provided for combined chart")

        self._set_style()

        product = data_list[0]["product_name"]
        market = data_list[0]["market_standard"]
        condition = data_list[0]["test_condition"]

        all_test_items = set()
        for data in data_list:
            all_test_items.update(data["test_items"].keys())
        all_test_items = sorted(list(all_test_items))

        time_axis = self._build_time_axis(data_list)
        x = np.arange(len(time_axis))

        n_items = len(all_test_items)
        fig, axes = plt.subplots(n_items, 1, figsize=self._figure_size(n_items), sharex=True)
        if n_items == 1:
            axes = [axes]

        palette = self._get_palette()
        for batch_idx, data in enumerate(data_list):
            batch = data["batch_number"]
            time_points = [str(tp) for tp in data["time_points"]]
            time_index = {tp: i for i, tp in enumerate(time_points)}

            for item_idx, item_name in enumerate(all_test_items):
                ax = axes[item_idx]
                values = data["test_items"].get(item_name)
                if not values:
                    continue

                aligned = []
                for label in time_axis:
                    if label in time_index and time_index[label] < len(values):
                        aligned.append(values[time_index[label]])
                    else:
                        aligned.append(None)
                y = np.array([v if v is not None else np.nan for v in aligned], dtype=float)
                color = palette[batch_idx % len(palette)]
                ax.plot(x, y, marker="o", label=batch, color=color)
                ax.set_title(item_name, loc="left", fontweight="bold")
                ax.set_ylabel(item_name)

        for item_idx, item_name in enumerate(all_test_items):
            ax = axes[item_idx]
            limits = None
            for data in data_list:
                candidate = data.get("test_limits", {}).get(item_name)
                if candidate and (candidate.get("lower") is not None or candidate.get("upper") is not None):
                    limits = candidate
                    break
            if limits:
                self._apply_limits(ax, limits.get("lower"), limits.get("upper"))

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(time_axis, rotation=0)
        axes[-1].set_xlabel("时间点")

        title = f"{product} - 多批次对比\n{market} | {condition}"
        temp_humidity = data_list[0].get("temperature_humidity", "未说明")
        if temp_humidity != "未说明":
            title += f"\n{temp_humidity}"
        fig.suptitle(title, fontsize=CHART_TITLE_FONT_SIZE, fontweight="bold", y=0.98)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        product_safe = product.replace("/", "_").replace("\\", "_")
        market_safe = market.replace("/", "_").replace("\\", "_")
        condition_safe = condition.replace("/", "_").replace("\\", "_")
        filename = f"{product_safe}_{market_safe}_{condition_safe}_{len(data_list)}batches.png"
        filepath = output_dir / filename
        if save:
            fig.savefig(filepath, bbox_inches="tight")
            self.charts_generated.append(filepath)
            print(f"  ✓ Combined chart saved: {filepath.name}")
        plt.close(fig)
        return filepath

    def generate_all_charts(
        self,
        grouped_data: Dict[str, List[Dict]],
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """Generate charts for all grouped data"""
        if output_dir is None:
            output_dir = CHARTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        self.charts_generated = []

        for group_key, data_list in grouped_data.items():
            print(f"\nGenerating charts for group: {group_key}")

            if len(data_list) == 1:
                self.create_single_batch_chart(data_list[0], save=True, output_dir=output_dir)
            else:
                self.create_combined_chart(data_list, save=True, output_dir=output_dir)

        return self.charts_generated

    def save_chart_summary(self, output_dir: Optional[Path] = None) -> Path:
        """Save a summary of generated charts"""
        if output_dir is None:
            output_dir = CHARTS_DIR

        summary_path = output_dir / "chart_summary.txt"

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Drug Chart Generation Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total charts generated: {len(self.charts_generated)}\n\n")
            f.write("Generated charts:\n")
            for chart_path in self.charts_generated:
                f.write(f"  - {chart_path.name}\n")

        print(f"\n✓ Chart summary saved: {summary_path.name}")
        return summary_path
