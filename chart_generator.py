"""
Chart generator module for creating interactive stability trend charts
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from config import (
    CHART_WIDTH,
    CHART_TITLE_FONT_SIZE,
    CHART_AXIS_FONT_SIZE,
    CHART_LEGEND_FONT_SIZE,
    CHARTS_DIR,
)


class ChartGenerator:
    """Generate interactive HTML charts for stability data"""

    def __init__(self):
        self.charts_generated = []

    @staticmethod
    def _apply_professional_style(
        fig: go.Figure,
        n_rows: int,
        show_legend: bool
    ) -> None:
        """Apply a consistent professional styling to a Plotly figure"""
        height_per_row = 320
        min_height = 520
        fig.update_layout(
            template="plotly_white",
            font=dict(
                family="Helvetica, Arial, sans-serif",
                size=CHART_AXIS_FONT_SIZE,
                color="#1f2937",
            ),
            width=CHART_WIDTH,
            height=max(min_height, height_per_row * n_rows),
            margin=dict(l=80, r=40, t=90, b=70),
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="white",
                bordercolor="rgba(0,0,0,0.15)",
                font=dict(color="#111827"),
            ),
            showlegend=show_legend,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=CHART_LEGEND_FONT_SIZE),
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="rgba(0,0,0,0.08)",
                borderwidth=1,
            ),
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            ticks="outside",
            tickcolor="rgba(0,0,0,0.2)",
            title_font=dict(size=CHART_AXIS_FONT_SIZE),
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.08)",
            zeroline=False,
            ticks="outside",
            tickcolor="rgba(0,0,0,0.2)",
            title_font=dict(size=CHART_AXIS_FONT_SIZE),
        )

    @staticmethod
    def _get_test_item_color(test_item: str, item_index: int) -> str:
        """Get color for a test item based on index

        Args:
            test_item: Test item name
            item_index: Index of test item

        Returns:
            Color string
        """
        # Use color based on item index
        colors = px.colors.qualitative.Set2
        return colors[item_index % len(colors)]

    @staticmethod
    def _get_batch_palette() -> Sequence[str]:
        return [
            "#2563eb", "#16a34a", "#f59e0b", "#ef4444", "#8b5cf6",
            "#0ea5e9", "#f97316", "#14b8a6", "#db2777", "#64748b",
        ]

    def create_single_batch_chart(
        self,
        data: Dict,
        save: bool = True,
        output_dir: Optional[Path] = None
    ) -> go.Figure:
        """Create a chart for a single batch

        Args:
            data: Cleaned stability data dictionary
            save: Whether to save chart to file
            output_dir: Output directory (uses CHARTS_DIR by default)

        Returns:
            Plotly Figure object
        """
        if output_dir is None:
            output_dir = CHARTS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get data
        product = data["product_name"]
        batch = data["batch_number"]
        market = data["market_standard"]
        condition = data["test_condition"]
        time_points = data["time_points"]
        test_items = data["test_items"]

        # Create subplots - one for each test item
        n_items = len(test_items)
        fig = make_subplots(
            rows=n_items,
            cols=1,
            subplot_titles=list(test_items.keys()),
            vertical_spacing=0.08,
        )

        # Add traces for each test item
        for idx, (item_name, values) in enumerate(test_items.items()):
            row = idx + 1

            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=values,
                    mode="lines+markers",
                    name=f"{batch} - {item_name}",
                    line=dict(color=self._get_test_item_color(item_name, idx), width=2.5),
                    marker=dict(size=7, line=dict(width=1, color="white")),
                    hovertemplate="%{x}<br>%{y:.3f}<extra></extra>",
                ),
                row=row,
                col=1,
            )

            # Update y-axis label
            fig.update_yaxes(title_text=item_name, row=row, col=1)

        # Update layout
        title = f"{product} - {batch}<br>{market} | {condition}"
        if data.get("temperature_humidity", "未说明") != "未说明":
            title += f"<br>{data['temperature_humidity']}"

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=CHART_TITLE_FONT_SIZE, color="#111827"),
                x=0.5,
                xanchor="center",
            ),
        )
        self._apply_professional_style(fig, n_items, show_legend=False)

        # Update x-axis
        fig.update_xaxes(title_text="时间点", row=n_items, col=1)

        # Save if requested
        if save:
            filename = f"{product}_{batch}.html"
            filepath = output_dir / filename
            fig.write_html(str(filepath), include_plotlyjs="cdn")
            self.charts_generated.append(filepath)
            print(f"  ✓ Chart saved: {filepath.name}")

        return fig

    def create_combined_chart(
        self,
        data_list: List[Dict],
        save: bool = True,
        output_dir: Optional[Path] = None
    ) -> go.Figure:
        """Create a combined chart for multiple batches with same grouping

        Args:
            data_list: List of cleaned data dictionaries
            save: Whether to save chart to file
            output_dir: Output directory (uses CHARTS_DIR by default)

        Returns:
            Plotly Figure object
        """
        if output_dir is None:
            output_dir = CHARTS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        if not data_list:
            raise ValueError("No data provided for combined chart")

        # Get common data
        product = data_list[0]["product_name"]
        market = data_list[0]["market_standard"]
        condition = data_list[0]["test_condition"]

        # Collect all unique test items across all batches
        all_test_items = set()
        for data in data_list:
            all_test_items.update(data["test_items"].keys())
        all_test_items = sorted(list(all_test_items))

        # Create subplots - one for each test item
        n_items = len(all_test_items)
        fig = make_subplots(
            rows=n_items,
            cols=1,
            subplot_titles=all_test_items,
            vertical_spacing=0.08,
        )

        # Add traces for each batch
        batch_color_idx = 0
        for data in data_list:
            batch = data["batch_number"]
            test_items = data["test_items"]
            time_points = data["time_points"]

            palette = self._get_batch_palette()
            batch_color = palette[batch_color_idx % len(palette)]

            for idx, item_name in enumerate(all_test_items):
                row = idx + 1

                if item_name in test_items:
                    values = test_items[item_name]

                    fig.add_trace(
                        go.Scatter(
                            x=time_points,
                            y=values,
                            mode="lines+markers",
                            name=f"{batch}",
                            legendgroup=batch,
                            showlegend=(idx == 0),
                            line=dict(color=batch_color, width=2.5),
                            marker=dict(size=7, line=dict(width=1, color="white")),
                            hovertemplate=f"{batch}<br>%{{x}}<br>%{{y:.3f}}<extra></extra>",
                        ),
                        row=row,
                        col=1,
                    )

            batch_color_idx += 1

        # Update y-axis labels
        for idx, item_name in enumerate(all_test_items):
            fig.update_yaxes(title_text=item_name, row=idx + 1, col=1)

        # Update layout
        title = f"{product} - 多批次对比<br>{market} | {condition}"
        temp_humidity = data_list[0].get("temperature_humidity", "未说明")
        if temp_humidity != "未说明":
            title += f"<br>{temp_humidity}"

        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=CHART_TITLE_FONT_SIZE, color="#111827"),
                x=0.5,
                xanchor="center",
            ),
        )
        self._apply_professional_style(fig, n_items, show_legend=True)

        # Update x-axis
        fig.update_xaxes(title_text="时间点", row=n_items, col=1)

        # Save if requested
        if save:
            # Create filename from product and condition
            product_safe = product.replace("/", "_").replace("\\", "_")
            market_safe = market.replace("/", "_").replace("\\", "_")
            condition_safe = condition.replace("/", "_").replace("\\", "_")
            batches_str = "_".join([d["batch_number"] for d in data_list])
            filename = f"{product_safe}_{market_safe}_{condition_safe}_{len(data_list)}batches.html"
            filepath = output_dir / filename
            fig.write_html(str(filepath), include_plotlyjs="cdn")
            self.charts_generated.append(filepath)
            print(f"  ✓ Combined chart saved: {filepath.name}")

        return fig

    def generate_all_charts(
        self,
        grouped_data: Dict[str, List[Dict]],
        output_dir: Optional[Path] = None
    ) -> List[Path]:
        """Generate charts for all grouped data

        Args:
            grouped_data: Dictionary of grouped data from data_extractor
            output_dir: Output directory (uses CHARTS_DIR by default)

        Returns:
            List of paths to generated chart files
        """
        if output_dir is None:
            output_dir = CHARTS_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        self.charts_generated = []

        for group_key, data_list in grouped_data.items():
            print(f"\nGenerating charts for group: {group_key}")

            if len(data_list) == 1:
                # Single batch chart
                self.create_single_batch_chart(data_list[0], save=True, output_dir=output_dir)
            else:
                # Combined chart for multiple batches
                self.create_combined_chart(data_list, save=True, output_dir=output_dir)

        return self.charts_generated

    def save_chart_summary(self, output_dir: Optional[Path] = None) -> Path:
        """Save a summary of generated charts

        Args:
            output_dir: Output directory (uses CHARTS_DIR by default)

        Returns:
            Path to summary file
        """
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
