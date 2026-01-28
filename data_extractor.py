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
    def _to_float(value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            cleaned = value.strip().replace(",", "")
            if not cleaned:
                return None
            match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    return None
        return None

    @staticmethod
    def _normalize_month(value) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            match = re.search(r"\d+", value)
            if match:
                return int(match.group())
        return None

    def _normalize_timepoints(self, timepoints: List) -> List[int]:
        months = []
        for tp in timepoints or []:
            month = self._normalize_month(tp)
            if month is not None:
                months.append(month)
        return sorted(set(months))

    def _normalize_range(self, payload) -> Dict:
        normalized = {"nominal": None, "tolerance": None, "min": None, "max": None, "raw": None}
        if isinstance(payload, dict):
            normalized["nominal"] = self._to_float(payload.get("nominal"))
            normalized["tolerance"] = self._to_float(payload.get("tolerance"))
            normalized["min"] = self._to_float(payload.get("min"))
            normalized["max"] = self._to_float(payload.get("max"))
            raw = payload.get("raw")
            if raw is not None and str(raw).strip():
                normalized["raw"] = str(raw).strip()
        elif payload is not None:
            normalized["raw"] = str(payload).strip()
        return normalized

    def _normalize_spec(self, spec) -> Dict:
        normalized = {"type": None, "min": None, "max": None, "value": None, "raw": None}
        if isinstance(spec, dict):
            normalized["type"] = spec.get("type")
            normalized["min"] = self._to_float(spec.get("min"))
            normalized["max"] = self._to_float(spec.get("max"))
            normalized["value"] = self._to_float(spec.get("value"))
            raw = spec.get("raw")
            if raw is not None and str(raw).strip():
                normalized["raw"] = str(raw).strip()
        elif spec is not None:
            normalized["raw"] = str(spec).strip()
        return normalized

    def _normalize_detection_limit(self, detection_limit) -> Dict:
        normalized = {"type": None, "value": None, "unit": None, "raw": None}
        if isinstance(detection_limit, dict):
            normalized["type"] = detection_limit.get("type")
            normalized["value"] = self._to_float(detection_limit.get("value"))
            unit = detection_limit.get("unit")
            if isinstance(unit, str) and unit.strip():
                normalized["unit"] = unit.strip()
            raw = detection_limit.get("raw")
            if raw is not None and str(raw).strip():
                normalized["raw"] = str(raw).strip()
        elif detection_limit is not None:
            normalized["raw"] = str(detection_limit).strip()
        return normalized

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
        return DataExtractor._to_float(value) is not None

    def validate_test_item(self, item_name: str, results: List[Dict]) -> bool:
        """Validate a test item and its results

        Args:
            item_name: Name of the test item
            results: List of results_by_timepoint dictionaries

        Returns:
            True if valid test item with numeric values, False otherwise
        """
        item_lower = (item_name or "").lower().strip()

        for rejected in REJECTED_TEST_ITEMS:
            if rejected.lower() in item_lower:
                return False

        is_accepted = any(accepted.lower() in item_lower for accepted in ACCEPTED_TEST_ITEMS)

        if not results:
            return False

        has_numeric = False
        for result in results:
            if not isinstance(result, dict):
                continue
            value = result.get("value")
            if value is None:
                value = self._to_float(result.get("raw"))
            if value is None:
                replicate_values = result.get("replicate_values") or []
                for rv in replicate_values:
                    if self._to_float(rv) is not None:
                        value = 0.0
                        break
            if value is not None:
                has_numeric = True

        if not is_accepted and not has_numeric:
            print(f"  Warning: '{item_name}' has no numeric results")

        return has_numeric

    def validate_and_clean_data(self, data: Dict) -> Optional[List[Dict]]:
        """Validate and clean extracted stability data (new schema)

        Args:
            data: Raw data from LLM (document-level object)

        Returns:
            List of cleaned study-level dictionaries, or None if invalid
        """
        if not isinstance(data, dict):
            print("  Error: Invalid data format")
            return None

        file_level = data.get("file_level", {}) if isinstance(data.get("file_level"), dict) else {}
        product_name = (file_level.get("product_name") or "").strip()
        if not product_name:
            product_name = "未知产品"

        file_regulatory = (file_level.get("regulatory_context") or "").strip()

        batches = data.get("batches", [])
        if not isinstance(batches, list) or not batches:
            print("  Error: Missing or invalid batches")
            return None

        cleaned_entries: List[Dict] = []

        for batch in batches:
            if not isinstance(batch, dict):
                continue
            batch_id = (batch.get("batch_id") or "").strip()
            if not batch_id:
                batch_id = "未知批号"
            elif not self.validate_batch_number(batch_id):
                print(f"  Warning: Invalid batch number format: {batch_id}")

            studies = batch.get("studies", [])
            if not isinstance(studies, list) or not studies:
                continue

            for study in studies:
                if not isinstance(study, dict):
                    continue

                regulatory_context = (study.get("regulatory_context") or file_regulatory or "未知").strip()
                stability_condition = (study.get("stability_condition") or "").strip() or None
                stability_condition_label = (study.get("stability_condition_label") or "").strip() or None
                condition_enum = (study.get("condition_enum") or "").strip() or None

                temperature_c = self._normalize_range(study.get("temperature_c"))
                humidity_rh = self._normalize_range(study.get("humidity_rh"))

                timepoints = self._normalize_timepoints(study.get("timepoints_months", []))

                items_raw = study.get("items", [])
                if not isinstance(items_raw, list):
                    items_raw = []

                cleaned_items = []
                derived_months = set()

                for item in items_raw:
                    if not isinstance(item, dict):
                        continue

                    item_name = item.get("item_name")
                    normalized_name = item.get("normalized_name")
                    name_for_validation = normalized_name or item_name or ""
                    unit = item.get("unit")
                    if isinstance(unit, str):
                        unit = unit.strip() or None

                    spec = self._normalize_spec(item.get("spec"))

                    results_raw = item.get("results_by_timepoint", [])
                    if not isinstance(results_raw, list):
                        results_raw = []

                    normalized_results = []
                    for result in results_raw:
                        if not isinstance(result, dict):
                            continue
                        month = self._normalize_month(result.get("month"))
                        if month is None:
                            month = self._normalize_month(result.get("raw"))
                        if month is not None:
                            derived_months.add(month)

                        value = self._to_float(result.get("value"))
                        raw = result.get("raw")
                        if raw is not None and str(raw).strip():
                            raw = str(raw).strip()

                        qualifier = result.get("qualifier")

                        replicate_values = result.get("replicate_values")
                        if isinstance(replicate_values, list):
                            parsed_reps = [self._to_float(rv) for rv in replicate_values]
                            replicate_values = [rv for rv in parsed_reps if rv is not None] or None

                        detection_limit = self._normalize_detection_limit(result.get("detection_limit"))

                        normalized_results.append(
                            {
                                "month": month,
                                "value": value,
                                "raw": raw,
                                "qualifier": qualifier,
                                "replicate_values": replicate_values,
                                "detection_limit": detection_limit,
                            }
                        )

                    if not self.validate_test_item(name_for_validation, normalized_results):
                        continue

                    cleaned_items.append(
                        {
                            "item_name": item_name,
                            "normalized_name": normalized_name,
                            "unit": unit,
                            "spec": spec,
                            "results_by_timepoint": normalized_results,
                            "confidence": item.get("confidence"),
                        }
                    )

                if not cleaned_items:
                    print("  Error: No valid test items found")
                    continue

                if not timepoints:
                    timepoints = sorted(derived_months)
                if not timepoints:
                    print("  Error: Missing timepoints_months")
                    continue

                cleaned_entry = {
                    "product_name": product_name,
                    "regulatory_context": regulatory_context or "未知",
                    "batch_id": batch_id,
                    "stability_condition": stability_condition,
                    "stability_condition_label": stability_condition_label,
                    "condition_enum": condition_enum,
                    "temperature_c": temperature_c,
                    "humidity_rh": humidity_rh,
                    "timepoints_months": timepoints,
                    "items": cleaned_items,
                    "source_snippets": study.get("source_snippets", []),
                    "confidence": study.get("confidence"),
                }

                if "_metadata" in data:
                    cleaned_entry["_metadata"] = data["_metadata"]

                cleaned_entries.append(cleaned_entry)

        if not cleaned_entries:
            print("  Error: No valid studies found")
            return None

        return cleaned_entries

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
        """Group data by (product, regulatory_context, stability_condition_label)

        Args:
            data_list: List of cleaned data dictionaries

        Returns:
            Dictionary with grouping keys and list of data
        """
        grouped = defaultdict(list)

        for data in data_list:
            condition = (
                data.get("stability_condition_label")
                or data.get("stability_condition")
                or data.get("condition_enum")
                or "未知条件"
            )
            key = f"{data.get('product_name', 'Unknown')}_{data.get('regulatory_context', '未知')}_{condition}"
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

        # Create filename from product, batch, and condition
        def _safe(value: str) -> str:
            return str(value).replace("/", "_").replace("\\", "_").strip()

        product = _safe(data.get("product_name", "Unknown"))
        batch = _safe(data.get("batch_id", "Unknown"))
        condition = _safe(
            data.get("stability_condition_label")
            or data.get("stability_condition")
            or data.get("condition_enum")
            or "condition"
        )
        source_stem = None
        source_file = data.get("_metadata", {}).get("source_file") if isinstance(data.get("_metadata"), dict) else None
        if source_file:
            try:
                source_stem = Path(source_file).stem
            except Exception:
                source_stem = None

        filename = f"{product}_{batch}_{condition}"
        if source_stem:
            filename += f"_{_safe(source_stem)}"
        filename += ".json"
        filepath = output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return filepath

    def process_batch(
        self,
        raw_data_list: List[Dict],
        output_dir: Optional[Path] = None,
    ) -> Tuple[List[Dict], List[Dict]]:
        """Process a batch of raw data from LLM (document-level objects)

        Args:
            raw_data_list: List of raw data dictionaries from LLM

        Returns:
            Tuple of (valid_data, invalid_data)
        """
        valid_data: List[Dict] = []
        invalid_data: List[Dict] = []

        for raw_data in raw_data_list:
            file_level = raw_data.get("file_level", {}) if isinstance(raw_data, dict) else {}
            product = file_level.get("product_name", "Unknown")
            batch_count = len(raw_data.get("batches", [])) if isinstance(raw_data, dict) else 0
            print(f"Processing: {product} - {batch_count} batch(es)")

            cleaned_list = self.validate_and_clean_data(raw_data)

            if cleaned_list:
                for cleaned in cleaned_list:
                    valid_data.append(cleaned)
                    filepath = self.save_extracted_data(cleaned, output_dir=output_dir)
                    print(f"  ✓ Saved: {filepath.name}")
            else:
                invalid_data.append(raw_data)
                print("  ✗ Invalid data")

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
            "unique_products": len(set(d.get("product_name") for d in self.extracted_data)),
            "unique_batches": len(set(d.get("batch_id") for d in self.extracted_data)),
        }
