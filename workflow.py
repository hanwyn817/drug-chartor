"""
Main workflow orchestrator for drug-chartor project
"""

from pathlib import Path
from typing import Optional, List, Union

from office_document_processor import process_documents_batch, iter_office_files
from llm_analyzer import LLMAnalyzer
from data_extractor import DataExtractor
from chart_generator import ChartGenerator

from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    EXTRACTED_DIR,
    CHARTS_DIR,
    ensure_directories,
    validate_config,
    get_relative_path,
)


class DrugChartWorkflow:
    """Main workflow for processing drug stability data and generating charts"""

    def __init__(
        self,
        input_dir: Union[str, Path, None] = None,
        output_dir: Union[str, Path, None] = None,
        extracted_dir: Union[str, Path, None] = None,
        charts_dir: Union[str, Path, None] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize workflow

        Args:
            input_dir: Input directory with raw documents
            output_dir: Output directory for converted files
            extracted_dir: Directory for extracted JSON data
            charts_dir: Directory for generated charts
            api_key: LLM API key
            base_url: LLM API base URL
            model: LLM model name
        """
        self.input_dir = Path(input_dir) if input_dir else INPUT_DIR
        self.output_dir = Path(output_dir) if output_dir else OUTPUT_DIR
        self.extracted_dir = Path(extracted_dir) if extracted_dir else EXTRACTED_DIR
        self.charts_dir = Path(charts_dir) if charts_dir else CHARTS_DIR

        # Initialize components
        self.llm_analyzer = LLMAnalyzer(api_key=api_key, base_url=base_url, model=model)
        self.data_extractor = DataExtractor()
        self.chart_generator = ChartGenerator()

        # Statistics
        self.stats = {
            "files_processed": 0,
            "files_with_stability_data": 0,
            "charts_generated": 0,
        }

    def _validate_input(self, skip_conversion: bool = False) -> bool:
        """Validate input directory exists and contains files

        Returns:
            True if valid, raises exception otherwise
        """
        if skip_conversion:
            existing_csv = list(self.output_dir.rglob("*.csv"))
            existing_html = list(self.output_dir.rglob("*.html"))
            if existing_csv or existing_html:
                return True

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        office_files = list(iter_office_files(self.input_dir))
        if not office_files:
            raise ValueError(f"No Office files found in: {self.input_dir}")
        return True

    def _convert_documents(self, skip_conversion: bool = False) -> List[Path]:
        """Convert Office documents to HTML/CSV

        Args:
            skip_conversion: Skip conversion if files already exist

        Returns:
            List of converted file paths
        """
        print("\n" + "="*80)
        print("Step 1: Converting Office documents")
        print("="*80)

        # Check if output directory already has converted files
        existing_csv = list(self.output_dir.rglob("*.csv"))
        existing_html = list(self.output_dir.rglob("*.html"))

        if skip_conversion and (existing_csv or existing_html):
            print("⚠ Skipping document conversion (files already exist)")
            print(f"   Found {len(existing_csv)} CSV files")
            print(f"   Found {len(existing_html)} HTML files")

            # Collect existing converted files
            converted_files = []
            for ext in [".csv", ".html"]:
                converted_files.extend(self.output_dir.rglob(f"*{ext}"))

            print(f"\n✓ Using {len(converted_files)} existing converted files")
            return converted_files

        # Process documents
        process_documents_batch(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            export_format="both",
            export_excel_csv=True,
            export_word_html=True,
        )

        # Collect converted files
        converted_files = []
        for ext in [".csv", ".html"]:
            converted_files.extend(self.output_dir.rglob(f"*{ext}"))

        print(f"\n✓ Converted {len(converted_files)} files to CSV/HTML format")
        return converted_files

    def _analyze_with_llm(self, converted_files: List[Path]) -> List[dict]:
        """Analyze converted files with LLM

        Args:
            converted_files: List of converted file paths

        Returns:
            List of extracted data dictionaries
        """
        print("\n" + "="*80)
        print("Step 2: Analyzing files with LLM")
        print("="*80)

        # Filter files
        valid_files = [f for f in converted_files if f.is_file() and f.suffix in [".csv", ".html"]]
        print(f"Found {len(valid_files)} files to analyze\n")

        if not valid_files:
            print("⚠ No valid files found for analysis")
            return []

        print("Analyzing files (this may take a few minutes)...\n")

        # Analyze files
        raw_data = self.llm_analyzer.batch_analyze_files(valid_files)

        self.stats["files_processed"] = len(valid_files)
        self.stats["files_with_stability_data"] = len(raw_data)

        print(f"\n✓ Found stability data in {len(raw_data)} files")

        return raw_data

    def _extract_and_validate(self, raw_data: List[dict]) -> tuple:
        """Extract and validate LLM data

        Args:
            raw_data: List of raw data from LLM

        Returns:
            Tuple of (valid_data, grouped_data)
        """
        print(f"\n{'='*80}")
        print("Step 3: Extracting and validating data")
        print(f"{'='*80}")

        # Process raw data
        valid_data, invalid_data = self.data_extractor.process_batch(
            raw_data,
            output_dir=self.extracted_dir,
        )

        # Print statistics
        stats = self.data_extractor.get_statistics()
        print(f"\nData processing statistics:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Valid: {stats['valid']}")
        print(f"  Invalid: {stats['invalid']}")
        print(f"  Unique products: {stats['unique_products']}")
        print(f"  Unique batches: {stats['unique_batches']}")

        # Group data
        grouped_data = self.data_extractor.aggregate_by_grouping(valid_data)

        print(f"\n✓ Grouped into {len(grouped_data)} chart groups:")
        for key, data_list in grouped_data.items():
            print(f"  - {key}: {len(data_list)} batch(es)")

        return valid_data, grouped_data

    def _generate_charts(self, grouped_data: dict) -> List[Path]:
        """Generate charts from grouped data

        Args:
            grouped_data: Dictionary of grouped data

        Returns:
            List of generated chart file paths
        """
        print(f"\n{'='*80}")
        print("Step 4: Generating charts")
        print(f"{'='*80}")

        # Generate all charts
        chart_files = self.chart_generator.generate_all_charts(grouped_data, self.charts_dir)

        # Save summary
        self.chart_generator.save_chart_summary(self.charts_dir)

        self.stats["charts_generated"] = len(chart_files)

        print(f"\n✓ Generated {len(chart_files)} interactive HTML charts")

        return chart_files

    def run(self, skip_conversion: bool = False) -> dict:
        """Run the complete workflow

        Args:
            skip_conversion: Skip document conversion if already done

        Returns:
            Dictionary with workflow statistics
        """
        # Validate configuration
        validate_config(self.llm_analyzer.api_key)
        ensure_directories([self.output_dir, self.extracted_dir, self.charts_dir])

        # Validate input
        self._validate_input(skip_conversion=skip_conversion)

        print(f"\n{'='*80}")
        print("Drug-Chartor Workflow Starting")
        print(f"{'='*80}")
        print(f"Input directory: {get_relative_path(self.input_dir)}")
        print(f"Output directory: {get_relative_path(self.output_dir)}")
        print(f"Extracted directory: {get_relative_path(self.extracted_dir)}")
        print(f"Charts directory: {get_relative_path(self.charts_dir)}")

        try:
            # Step 1: Convert documents
            converted_files = self._convert_documents(skip_conversion)

            # Step 2: Analyze with LLM
            raw_data = self._analyze_with_llm(converted_files)

            if not raw_data:
                print("\n⚠ No stability data found. Workflow completed.")
                return self.stats

            # Step 3: Extract and validate
            valid_data, grouped_data = self._extract_and_validate(raw_data)

            if not valid_data:
                print("\n⚠ No valid data after validation. Workflow completed.")
                return self.stats

            # Step 4: Generate charts
            chart_files = self._generate_charts(grouped_data)

        except Exception as e:
            print(f"\n❌ Error during workflow: {e}")
            raise

        # Print summary
        print(f"\n{'='*80}")
        print("Workflow Summary")
        print(f"{'='*80}")
        print(f"Files processed: {self.stats['files_processed']}")
        print(f"Files with stability data: {self.stats['files_with_stability_data']}")
        print(f"Charts generated: {self.stats['charts_generated']}")
        print(f"\n✓ Workflow completed successfully!")

        return self.stats


def main_workflow(
    input_dir: str = "./input",
    output_dir: str = "./output",
    extracted_dir: str = "./extracted",
    charts_dir: str = "./charts",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    skip_conversion: bool = False,
) -> dict:
    """Convenience function to run workflow with parameters

    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        extracted_dir: Extracted data directory path
        charts_dir: Charts output directory path
        api_key: LLM API key
        base_url: LLM API base URL
        model: LLM model name
        skip_conversion: Skip document conversion

    Returns:
        Dictionary with workflow statistics
    """
    workflow = DrugChartWorkflow(
        input_dir=input_dir,
        output_dir=output_dir,
        extracted_dir=extracted_dir,
        charts_dir=charts_dir,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )

    return workflow.run(skip_conversion=skip_conversion)
