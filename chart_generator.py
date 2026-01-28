"""
Chart generator module for creating static stability trend charts
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

import matplotlib.pyplot as plt
from matplotlib import font_manager as font_manager
import numpy as np

from config import (
    CHART_WIDTH,
    CHART_HEIGHT,
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
        preferred_fonts = ChartGenerator._pick_cjk_fonts()
        if not preferred_fonts:
            preferred_fonts = [
                "PingFang SC",
                "Hiragino Sans GB",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Microsoft YaHei",
                "SimHei",
                "WenQuanYi Zen Hei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ]
        plt.rcParams.update(
            {
                "figure.dpi": 150,
                "savefig.dpi": 300,
                "font.size": CHART_AXIS_FONT_SIZE,
                "axes.titlesize": CHART_AXIS_FONT_SIZE + 2,
                "axes.labelsize": CHART_AXIS_FONT_SIZE,
                "legend.fontsize": CHART_LEGEND_FONT_SIZE,
                "font.family": "sans-serif",
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
                "axes.unicode_minus": False,
                # Font fallback for CJK
                "font.sans-serif": preferred_fonts,
            }
        )

    @staticmethod
    def _pick_cjk_fonts() -> List[str]:
        preferred = [
            "PingFang SC",
            "Hiragino Sans GB",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "Microsoft YaHei",
            "SimHei",
            "WenQuanYi Zen Hei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        fonts = font_manager.fontManager.ttflist
        available = {font.name for font in fonts}
        ordered = [name for name in preferred if name in available]
        if ordered:
            return ordered

        keywords = [
            "pingfang",
            "hiragino",
            "notosanscjk",
            "noto sans cjk",
            "source han sans",
            "microsoft yahei",
            "simhei",
            "wenquanyi",
            "arial unicode",
        ]
        matched = []
        seen = set()
        for font in fonts:
            haystack = f"{font.name} {font.fname}".lower()
            if any(keyword in haystack for keyword in keywords):
                if font.name not in seen:
                    matched.append(font.name)
                    seen.add(font.name)
        return matched

    @staticmethod
    def _normalize_month(value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        match = re.search(r"\d+", str(value))
        if match:
            return int(match.group())
        return None

    @staticmethod
    def _format_month_label(month: Optional[int]) -> str:
        if month is None:
            return "未知"
        return f"{month}月"

    def _extract_timepoints(self, data: Dict) -> List[int]:
        months = []
        for tp in data.get("timepoints_months", []):
            month = self._normalize_month(tp)
            if month is not None:
                months.append(month)
        if not months:
            for item in data.get("items", []):
                for result in item.get("results_by_timepoint", []):
                    month = self._normalize_month(result.get("month"))
                    if month is not None:
                        months.append(month)
        return sorted(set(months))

    def _build_time_axis(self, data_list: List[Dict]) -> List[int]:
        months = set()
        for data in data_list:
            for month in self._extract_timepoints(data):
                months.add(month)
        return sorted(months)

    @staticmethod
    def _item_label(item: Dict) -> str:
        name = item.get("item_name") or item.get("normalized_name")
        return str(name) if name else "未知项目"

    @staticmethod
    def _spec_to_limits(spec: Optional[Dict]) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(spec, dict):
            return None, None
        lower = spec.get("min")
        upper = spec.get("max")
        value = spec.get("value")
        spec_type = str(spec.get("type") or "").lower()
        if lower is None and upper is None and value is not None:
            if spec_type in {"max", "upper", "<=", "≤", "not_more_than"}:
                upper = value
            elif spec_type in {"min", "lower", ">=", "≥", "not_less_than"}:
                lower = value
        return lower, upper

    @staticmethod
    def _format_temperature(temp: Optional[Dict]) -> Optional[str]:
        if not isinstance(temp, dict):
            return str(temp) if temp else None
        if temp.get("raw"):
            return str(temp.get("raw"))
        if temp.get("min") is not None and temp.get("max") is not None:
            return f"{temp['min']}-{temp['max']}℃"
        if temp.get("nominal") is not None:
            if temp.get("tolerance") is not None:
                return f"{temp['nominal']}±{temp['tolerance']}℃"
            return f"{temp['nominal']}℃"
        return None

    @staticmethod
    def _format_humidity(humidity: Optional[Dict]) -> Optional[str]:
        if not isinstance(humidity, dict):
            return str(humidity) if humidity else None
        if humidity.get("raw"):
            return str(humidity.get("raw"))
        if humidity.get("min") is not None and humidity.get("max") is not None:
            return f"{humidity['min']}-{humidity['max']}%RH"
        if humidity.get("nominal") is not None:
            if humidity.get("tolerance") is not None:
                return f"{humidity['nominal']}±{humidity['tolerance']}%RH"
            return f"{humidity['nominal']}%RH"
        return None

    def _format_environment(self, data: Dict) -> Optional[str]:
        temp = self._format_temperature(data.get("temperature_c"))
        humidity = self._format_humidity(data.get("humidity_rh"))
        if temp and humidity:
            return f"{temp} / {humidity}"
        return temp or humidity

    @staticmethod
    def _get_palette() -> List[str]:
        return [
            "#2563eb", "#16a34a", "#f59e0b", "#ef4444", "#8b5cf6",
            "#0ea5e9", "#f97316", "#14b8a6", "#db2777", "#64748b",
        ]

    @staticmethod
    def _figure_size(n_rows: int) -> Tuple[float, float]:
        width_in = max(10.0, CHART_WIDTH / 100.0)
        base_height = max(5.5, CHART_HEIGHT / 100.0)
        if n_rows <= 1:
            height_in = base_height
        else:
            height_in = max(base_height, n_rows * 3.5)
        return width_in, height_in

    @staticmethod
    def _safe_filename(value: str) -> str:
        cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", str(value))
        cleaned = re.sub(r"\s+", "_", cleaned).strip("_")
        return cleaned or "unknown"

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
    ) -> List[Path]:
        """Create charts for each item in a single batch and save as PNG"""
        if output_dir is None:
            output_dir = CHARTS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)

        self._set_style()

        product = data.get("product_name", "Unknown")
        batch = data.get("batch_id", "Unknown")
        regulatory = data.get("regulatory_context", "未知")
        condition = (
            data.get("stability_condition_label")
            or data.get("stability_condition")
            or data.get("condition_enum")
            or "未知条件"
        )
        time_axis = self._build_time_axis([data])
        items = [item for item in data.get("items", []) if isinstance(item, dict)]

        if not items:
            return []

        chart_paths: List[Path] = []
        x = np.arange(len(time_axis))
        month_labels = [self._format_month_label(m) for m in time_axis]
        env = self._format_environment(data)

        for item in items:
            item_name = self._item_label(item)
            month_to_value = {}
            for result in item.get("results_by_timepoint", []):
                if not isinstance(result, dict):
                    continue
                month = self._normalize_month(result.get("month"))
                if month is None:
                    continue
                month_to_value[month] = result.get("value")

            values = [month_to_value.get(m) for m in time_axis]
            y = np.array([v if v is not None else np.nan for v in values], dtype=float)

            fig, ax = plt.subplots(1, 1, figsize=self._figure_size(1))
            ax.plot(x, y, marker="o", color="#2563eb")
            ax.set_title(item_name, loc="left", fontweight="bold")
            ax.set_ylabel(item_name)

            lower, upper = self._spec_to_limits(item.get("spec"))
            self._apply_limits(ax, lower, upper)

            ax.set_xticks(x)
            ax.set_xticklabels(month_labels, rotation=0)
            ax.set_xlabel("考察点位（月）")

            title = f"{product} - {batch}\n{regulatory} | {condition}"
            if env:
                title += f"\n{env}"
            fig.suptitle(title, fontsize=CHART_TITLE_FONT_SIZE, fontweight="bold", y=0.98)

            fig.tight_layout(rect=[0, 0, 1, 0.92])

            filename = f"{self._safe_filename(product)}_{self._safe_filename(batch)}_{self._safe_filename(item_name)}.png"
            filepath = output_dir / filename
            if save:
                fig.savefig(filepath, bbox_inches="tight")
                self.charts_generated.append(filepath)
                print(f"  ✓ Chart saved: {filepath.name}")
            plt.close(fig)
            chart_paths.append(filepath)

        return chart_paths

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

        product = data_list[0].get("product_name", "Unknown")
        regulatory = data_list[0].get("regulatory_context", "未知")
        condition = (
            data_list[0].get("stability_condition_label")
            or data_list[0].get("stability_condition")
            or data_list[0].get("condition_enum")
            or "未知条件"
        )

        all_test_items = []
        seen_items = set()
        for data in data_list:
            for item in data.get("items", []):
                name = self._item_label(item)
                if name not in seen_items:
                    seen_items.add(name)
                    all_test_items.append(name)

        time_axis = self._build_time_axis(data_list)
        x = np.arange(len(time_axis))

        n_items = len(all_test_items)
        fig, axes = plt.subplots(n_items, 1, figsize=self._figure_size(n_items), sharex=True)
        if n_items == 1:
            axes = [axes]

        palette = self._get_palette()
        for batch_idx, data in enumerate(data_list):
            batch = data.get("batch_id", "Unknown")
            item_map = {}
            for item in data.get("items", []):
                name = self._item_label(item)
                month_to_value = {}
                for result in item.get("results_by_timepoint", []):
                    if not isinstance(result, dict):
                        continue
                    month = self._normalize_month(result.get("month"))
                    if month is None:
                        continue
                    month_to_value[month] = result.get("value")
                item_map[name] = month_to_value

            for item_idx, item_name in enumerate(all_test_items):
                ax = axes[item_idx]
                month_to_value = item_map.get(item_name, {})
                aligned = [month_to_value.get(m) for m in time_axis]
                y = np.array([v if v is not None else np.nan for v in aligned], dtype=float)
                color = palette[batch_idx % len(palette)]
                ax.plot(x, y, marker="o", label=batch, color=color)
                ax.set_title(item_name, loc="left", fontweight="bold")
                ax.set_ylabel(item_name)

        for item_idx, item_name in enumerate(all_test_items):
            ax = axes[item_idx]
            limits = None
            for data in data_list:
                for item in data.get("items", []):
                    if self._item_label(item) != item_name:
                        continue
                    lower, upper = self._spec_to_limits(item.get("spec"))
                    if lower is not None or upper is not None:
                        limits = (lower, upper)
                        break
                if limits:
                    break
            if limits:
                self._apply_limits(ax, limits[0], limits[1])

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels([self._format_month_label(m) for m in time_axis], rotation=0)
        axes[-1].set_xlabel("考察点位（月）")

        title = f"{product} - 多批次对比\n{regulatory} | {condition}"
        env = self._format_environment(data_list[0])
        if env:
            title += f"\n{env}"
        fig.suptitle(title, fontsize=CHART_TITLE_FONT_SIZE, fontweight="bold", y=0.98)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.98, 0.98))

        fig.tight_layout(rect=[0, 0, 1, 0.94])

        product_safe = product.replace("/", "_").replace("\\", "_")
        regulatory_safe = regulatory.replace("/", "_").replace("\\", "_")
        condition_safe = condition.replace("/", "_").replace("\\", "_")
        filename = f"{product_safe}_{regulatory_safe}_{condition_safe}_{len(data_list)}batches.png"
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

            for data in data_list:
                self.create_single_batch_chart(data, save=True, output_dir=output_dir)

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
