"""
Data extractor module for validating and processing LLM-extracted stability data
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from config import (
    BATCH_NUMBER_PATTERN,
    ACCEPTED_TEST_ITEMS,
    REJECTED_TEST_ITEMS,
    EXTRACTED_DIR,
)


class DataExtractor:
    """Extract, validate, and process stability data from LLM output"""

    def __init__(self):
        self.extracted_data = []
        self.invalid_data = []

    @staticmethod
    def validate_batch_number(batch: str) -> bool:
        """Validate batch number format (D5XXX-YY-ZZZ)

        Args:
            batch: Batch number string

        Returns:
            True if valid format, False otherwise
        """
        pattern = re.compile(BATCH_NUMBER_PATTERN)
        return bool(pattern.match(batch))

    @staticmethod
    def is_numeric_value(value) -> bool:
        """Check if value is numeric (int or float)

        Args:
            value: Value to check

        Returns:
            True if numeric, False otherwise
        """
        if value is None:
            return False
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def validate_test_item(self, item_name: str, values: List) -> bool:
        """Validate a test item and its values

        Args:
            item_name: Name of the test item
            values: List of numeric values

        Returns:
            True if valid test item with numeric values, False otherwise
        """
        # Normalize item name to lowercase for comparison
        item_lower = item_name.lower().strip()

        # Check if explicitly rejected (text-only items)
        for rejected in REJECTED_TEST_ITEMS:
            if rejected.lower() in item_lower:
                return False

        # Check if it's an accepted test item type
        is_accepted = False
        for accepted in ACCEPTED_TEST_ITEMS:
            if accepted.lower() in item_lower:
                is_accepted = True
                break

        # If not in accepted list, still allow if it has numeric values
        # But must verify all values are numeric
        if not is_accepted:
            print(f"  Warning: '{item_name}' not in standard test items")

        # Check if all values are numeric (None is allowed for missing values)
        if not values:
            return False

        has_numeric = False
        for v in values:
            if v is None:
                continue
            if not self.is_numeric_value(v):
                print(f"  Warning: Non-numeric value in '{item_name}': {v}")
                return False
            has_numeric = True

        return has_numeric

    def validate_and_clean_data(self, data: Dict) -> Optional[Dict]:
        """Validate and clean extracted stability data

        Args:
            data: Raw data from LLM

        Returns:
            Cleaned data dictionary, or None if invalid
        """
        cleaned_data = {
            "product_name": "",
            "batch_number": "",
            "market_standard": "",
            "test_condition": "",
            "temperature_humidity": "",
            "time_points": [],
            "test_items": {},
            "test_limits": {},
        }

        # Validate product name
        product_name = data.get("product_name", "").strip()
        if not product_name:
            print("  Error: Missing product name")
            return None
        cleaned_data["product_name"] = product_name

        # Validate batch number
        batch_number = data.get("batch_number", "").strip()
        if not batch_number:
            print("  Error: Missing batch number")
            return None
        if not self.validate_batch_number(batch_number):
            print(f"  Error: Invalid batch number format: {batch_number}")
            return None
        cleaned_data["batch_number"] = batch_number

        # Market standard (optional, default to "未知" if missing)
        market_standard = data.get("market_standard", "").strip()
        if not market_standard:
            market_standard = "未知"
        cleaned_data["market_standard"] = market_standard

        # Test condition
        test_condition = data.get("test_condition", "").strip()
        if not test_condition:
            test_condition = "未知条件"
        cleaned_data["test_condition"] = test_condition

        # Temperature/humidity (optional)
        temp_humidity = data.get("temperature_humidity", "").strip()
        if not temp_humidity:
            temp_humidity = "未说明"
        cleaned_data["temperature_humidity"] = temp_humidity

        # Time points
        time_points = data.get("time_points", [])
        if not isinstance(time_points, list) or not time_points:
            print("  Error: Missing or invalid time_points")
            return None

        # Parse time points and extract numeric months
        cleaned_time_points = []
        for idx, tp in enumerate(time_points):
            if isinstance(tp, str):
                match = re.search(r"\d+", tp)
                if match:
                    month = int(match.group())
                    cleaned_time_points.append({"original": tp, "month": month, "index": idx})
                else:
                    print(f"  Warning: Cannot parse time point: {tp}")

        if not cleaned_time_points:
            print("  Error: No valid time points found")
            return None

        # Sort by month and keep original indices for value reordering
        cleaned_time_points.sort(key=lambda x: x["month"])
        cleaned_data["time_points"] = [tp["original"] for tp in cleaned_time_points]
        sorted_indices = [tp["index"] for tp in cleaned_time_points]

        # Test items
        test_items = data.get("test_items", {})
        if not isinstance(test_items, dict):
            print("  Error: Invalid test_items format")
            return None

        valid_items = {}
        valid_limits = {}
        for item_name, values in test_items.items():
            if not isinstance(item_name, str) or not isinstance(values, list):
                continue

            if not values:
                continue

            # Reorder values according to sorted time points
            reordered_values = []
            for idx in sorted_indices:
                if idx < len(values):
                    reordered_values.append(values[idx])
                else:
                    reordered_values.append(None)

            # Ensure values match number of time points
            if len(reordered_values) != len(cleaned_time_points):
                print(f"  Warning: '{item_name}' has {len(reordered_values)} values but {len(cleaned_time_points)} time points")
                if len(reordered_values) < len(cleaned_time_points):
                    reordered_values = reordered_values + [None] * (len(cleaned_time_points) - len(reordered_values))
                else:
                    reordered_values = reordered_values[:len(cleaned_time_points)]

            # Validate test item
            if self.validate_test_item(item_name, reordered_values):
                # Convert values to floats
                numeric_values = []
                for v in reordered_values:
                    if v is not None and self.is_numeric_value(v):
                        numeric_values.append(float(v))
                    else:
                        numeric_values.append(None)

                valid_items[item_name] = numeric_values

                # Normalize limits if provided
                raw_limits = {}
                limits_data = data.get("test_limits", {})
                if isinstance(limits_data, dict):
                    raw_limits = limits_data.get(item_name, {}) or {}
                lower = raw_limits.get("lower")
                upper = raw_limits.get("upper")
                norm_lower = float(lower) if self.is_numeric_value(lower) else None
                norm_upper = float(upper) if self.is_numeric_value(upper) else None
                if norm_lower is not None or norm_upper is not None:
                    valid_limits[item_name] = {"lower": norm_lower, "upper": norm_upper}

        if not valid_items:
            print("  Error: No valid test items found")
            return None

        cleaned_data["test_items"] = valid_items
        cleaned_data["test_limits"] = valid_limits

        # Preserve metadata
        if "_metadata" in data:
            cleaned_data["_metadata"] = data["_metadata"]

        return cleaned_data

    def parse_llm_response(self, response: str) -> Optional[Dict]:
        """Parse LLM JSON response

        Args:
            response: Raw response from LLM

        Returns:
            Parsed dictionary, or None if parsing failed
        """
        try:
            # Try to parse as JSON
            data = json.loads(response)
            return data
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    return data
                except json.JSONDecodeError:
                    return None
            return None

    def aggregate_by_grouping(self, data_list: List[Dict]) -> Dict[str, List[Dict]]:
        """Group data by (product, market_standard, test_condition)

        Args:
            data_list: List of cleaned data dictionaries

        Returns:
            Dictionary with grouping keys and list of data
        """
        grouped = defaultdict(list)

        for data in data_list:
            # Create grouping key
            key = f"{data['product_name']}_{data['market_standard']}_{data['test_condition']}"
            grouped[key].append(data)

        return dict(grouped)

    def save_extracted_data(self, data: Dict, output_dir: Optional[Path] = None) -> Path:
        """Save extracted data to JSON file

        Args:
            data: Cleaned data dictionary
            output_dir: Output directory (uses EXTRACTED_DIR by default)

        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = EXTRACTED_DIR

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename from product and batch
        product = data["product_name"].replace("/", "_").replace("\\", "_")
        batch = data["batch_number"]
        filename = f"{product}_{batch}.json"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return filepath

    def process_batch(
        self,
        raw_data_list: List[Dict],
        output_dir: Optional[Path] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process a batch of raw data from LLM

        Args:
            raw_data_list: List of raw data dictionaries from LLM

        Returns:
            Tuple of (valid_data, invalid_data)
        """
        valid_data = []
        invalid_data = []

        for raw_data in raw_data_list:
            print(f"Processing: {raw_data.get('product_name', 'Unknown')} - {raw_data.get('batch_number', 'Unknown')}")

            cleaned = self.validate_and_clean_data(raw_data)

            if cleaned:
                valid_data.append(cleaned)
                # Save to file
                filepath = self.save_extracted_data(cleaned, output_dir=output_dir)
                print(f"  ✓ Saved: {filepath.name}")
            else:
                invalid_data.append(raw_data)
                print(f"  ✗ Invalid data")

        self.extracted_data = valid_data
        self.invalid_data = invalid_data

        return valid_data, invalid_data

    def get_statistics(self) -> Dict[str, int]:
        """Get statistics of processed data

        Returns:
            Dictionary with statistics
        """
        return {
            "total_processed": len(self.extracted_data) + len(self.invalid_data),
            "valid": len(self.extracted_data),
            "invalid": len(self.invalid_data),
            "unique_products": len(set(d["product_name"] for d in self.extracted_data)),
            "unique_batches": len(set(d["batch_number"] for d in self.extracted_data)),
        }
