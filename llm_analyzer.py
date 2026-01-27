"""
LLM Analyzer module for extracting stability data
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from openai import OpenAI
from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_TIMEOUT,
    MAX_RETRIES,
    RETRY_DELAY,
)


class LLMAnalyzer:
    """Analyzer for extracting stability data using LLM"""

    SCREENING_PROMPT = """You are analyzing a pharmaceutical document. Determine if this document contains drug stability trend data.

Stability trend data includes:
- Test results at multiple time points (months: 0, 3, 6, 9, 12, 18, 24, 36, 48, etc.)
- Multiple test items with numeric values (moisture, related substances, assay, etc.)
- Batch numbers like D5XXX-YY-ZZZ (may have 1-2 extra letters/numbers suffix, e.g., D5XXX-YY-ZZZM1)

Examples of stability data:
- A table showing "0 month", "3 month", "6 month" columns with numeric results
- Multiple rows of test items (drying loss, moisture, impurities) over time
- Data tracking changes in drug quality over storage periods

Answer ONLY with "YES" or "NO" (no other text).
"""

    EXTRACTION_PROMPT = """Extract stability data from this pharmaceutical document.
If the document contains multiple batches, return ALL batches.

Return JSON in ONE of these formats:

Format A (preferred, multiple batches):
```json
[
  {
    "product_name": "string",
    "batch_number": "string (format: D5XXX-YY-ZZZ with optional 1-2 alphanumeric suffix)",
    "market_standard": "string",
    "test_condition": "string",
    "temperature_humidity": "string",
    "time_points": ["string (e.g., '0月', '3月')"],
    "test_items": {
      "test_item_name": [numeric_values],
      "test_item_name_2": [numeric_values]
    },
    "test_limits": {
      "test_item_name": {"lower": number_or_null, "upper": number_or_null},
      "test_item_name_2": {"lower": number_or_null, "upper": number_or_null}
    }
  }
]
```

Format B (single batch):
```json
{
  "product_name": "string",
  "batch_number": "string (format: D5XXX-YY-ZZZ with optional 1-2 alphanumeric suffix)",
  "market_standard": "string",
  "test_condition": "string",
  "temperature_humidity": "string",
  "time_points": ["string (e.g., '0月', '3月')"],
  "test_items": {
    "test_item_name": [numeric_values],
    "test_item_name_2": [numeric_values]
  },
  "test_limits": {
    "test_item_name": {"lower": number_or_null, "upper": number_or_null},
    "test_item_name_2": {"lower": number_or_null, "upper": number_or_null}
  }
}
```

Important rules:
1. product_name: Extract product name (e.g., 达格列净)
2. batch_number: Must match format D5XXX-YY-ZZZ with optional 1-2 alphanumeric suffix. If multiple batches, return all of them.
3. market_standard: e.g., CEP, EDMF, USDMF, 国内标准, 国内拟申报
4. test_condition: e.g., 长期Ⅱ, 长期ⅣB, 加速, 中间条件
5. temperature_humidity: e.g., 25℃/60%RH, 40℃/75%RH
6. time_points: List of time points in Chinese or English (e.g., ["0月", "3月", "6月"])
7. test_items: Dictionary with test item names as keys and lists of numeric values as values. ONLY include items with numeric values:
   - 干燥失重
   - 水分
   - 有关物质
   - 杂质
   - 含量/含量测定
   - related substances, impurities, assay, etc.
8. test_limits: Extract lower/upper limits for each test item if present in tables. If limits are not present in the document or are text-only (e.g., "报告值"), set them to null.

EXCLUDE text-only items like:
- 外观
- 红外
- 鉴别

IMPORTANT:
- Return ONLY the JSON (object or array), no markdown code blocks
- Ensure all numeric values are valid numbers (integers or floats)
- Ensure the number of values in each test item list matches the number of time_points
- If data cannot be extracted, return null
"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        """Initialize LLM analyzer

        Args:
            api_key: OpenAI API key (uses config default if not provided)
            base_url: API base URL (uses config default if not provided)
            model: Model name (uses config default if not provided)
        """
        self.api_key = api_key or LLM_API_KEY
        self.base_url = base_url or LLM_BASE_URL
        self.model = model or LLM_MODEL

        if not self.api_key:
            raise ValueError("API key is required")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _call_llm(self, prompt: str, content: str) -> str:
        """Call LLM with retry logic

        Args:
            prompt: System prompt
            content: User content (document text)

        Returns:
            LLM response text
        """
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": content[:20000]},  # Limit content length
                    ],
                    temperature=LLM_TEMPERATURE,
                    max_tokens=LLM_MAX_TOKENS,
                    timeout=LLM_TIMEOUT,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    raise Exception(f"LLM API call failed after {MAX_RETRIES} attempts: {e}")

    def is_stability_data(self, file_content: str) -> bool:
        """Check if file contains stability trend data

        Args:
            file_content: Text content of the file

        Returns:
            True if file contains stability data, False otherwise
        """
        try:
            response = self._call_llm(self.SCREENING_PROMPT, file_content)
            return response.upper() == "YES"
        except Exception as e:
            print(f"Error screening file: {e}")
            return False

    def _normalize_extracted_data(self, data: object) -> List[Dict]:
        if data is None:
            return []

        if isinstance(data, dict):
            return [data]

        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]

        return []

    def _is_valid_extracted(self, data: Dict) -> bool:
        required_fields = [
            "product_name",
            "batch_number",
            "market_standard",
            "test_condition",
            "time_points",
            "test_items",
        ]
        return all(field in data for field in required_fields)

    def extract_stability_data(self, file_content: str) -> Optional[List[Dict]]:
        """Extract structured stability data from file

        Args:
            file_content: Text content of the file

        Returns:
            List of extracted stability data dictionaries, or None if extraction failed
        """
        try:
            response = self._call_llm(self.EXTRACTION_PROMPT, file_content)

            # Clean response - remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            # Parse JSON
            data = json.loads(response)
            items = self._normalize_extracted_data(data)
            items = [item for item in items if self._is_valid_extracted(item)]
            return items or None

        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            print(f"Error extracting data: {e}")
            return None

    def analyze_file(self, file_path: Path) -> Optional[List[Dict]]:
        """Analyze a single file and extract stability data

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with extracted data and metadata, or None if no stability data
        """
        try:
            # Read file content
            if file_path.suffix.lower() == ".csv":
                content = file_path.read_text(encoding="utf-8-sig")
            else:
                content = file_path.read_text(encoding="utf-8")

            # Check if contains stability data
            if not self.is_stability_data(content):
                return None

            # Extract stability data
            items = self.extract_stability_data(content)
            if not items:
                return None

            # Add metadata
            for item in items:
                item["_metadata"] = {
                    "source_file": str(file_path),
                    "file_type": file_path.suffix.lower(),
                }

            return items

        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")
            return None

    def batch_analyze_files(self, file_paths: List[Path]) -> List[Dict]:
        """Analyze multiple files in batch

        Args:
            file_paths: List of file paths to analyze

        Returns:
            List of extracted data dictionaries
        """
        results = []
        for file_path in file_paths:
            print(f"Analyzing: {file_path.name}")
            result = self.analyze_file(file_path)
            if result:
                results.extend(result)
                if len(result) == 1:
                    item = result[0]
                    print(
                        f"  ✓ Extracted: {item.get('product_name', 'Unknown')} - "
                        f"{item.get('batch_number', 'Unknown')}"
                    )
                else:
                    print(f"  ✓ Extracted {len(result)} batches")
            else:
                print(f"  ✗ No stability data found")

        return results
