"""
Configuration module for drug-chartor project
"""

import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Directory paths
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
EXTRACTED_DIR = PROJECT_ROOT / "extracted"
CHARTS_DIR = PROJECT_ROOT / "charts"

# LLM API Configuration
LLM_API_KEY = os.getenv("OPENAI_API_KEY", "")
LLM_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# LLM Processing Settings
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 4000
LLM_TIMEOUT = 60

# Retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 2

# Batch processing
BATCH_SIZE = 5

# Stability data validation patterns
# Allow optional 1-2 alphanumeric suffix after base batch number
BATCH_NUMBER_PATTERN = r"^D5\d{3}-\d{2}-\d{3}[A-Za-z0-9]{0,2}$"

# Accepted test items (numeric values only)
ACCEPTED_TEST_ITEMS = {
    "干燥失重", "水分", "有关物质", "杂质", "含量",
    "loss on drying", "moisture", "related substances",
    "impurities", "assay", "potency"
}

# Rejected test items (text-only descriptions)
REJECTED_TEST_ITEMS = {
    "外观", "appearance", "红外", "infrared", "IR",
    "紫外", "UV", "鉴别", "identification"
}

# Common market/standard values
MARKET_STANDARDS = [
    "CEP", "EDMF", "USDMF", "国内标准", "国内拟申报",
    "DOMESTIC", "USA", "EU", "JP"
]

# Common test conditions
TEST_CONDITIONS = [
    "长期Ⅰ", "长期Ⅱ", "长期Ⅲ", "长期Ⅳ", "长期ⅣA", "长期ⅣB",
    "加速", "中间条件",
    "Long-term", "Accelerated", "Intermediate"
]

# Chart settings
CHART_WIDTH = 1200
CHART_HEIGHT = 800
CHART_TITLE_FONT_SIZE = 20
CHART_AXIS_FONT_SIZE = 14
CHART_LEGEND_FONT_SIZE = 12

# Color palette for multiple batches
BATCH_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
]


def ensure_directories(directories: Optional[List[Path]] = None) -> None:
    """Create all necessary directories if they don't exist"""
    if directories is None:
        directories = [OUTPUT_DIR, EXTRACTED_DIR, CHARTS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def validate_config(api_key: Optional[str] = None) -> bool:
    """Validate required configuration settings"""
    effective_key = api_key if api_key is not None else LLM_API_KEY
    if not effective_key:
        raise ValueError(
            "LLM_API_KEY not set. Please set OPENAI_API_KEY environment variable "
            "or provide it via --api-key parameter."
        )
    return True


def get_relative_path(absolute_path: Path) -> Path:
    """Get path relative to project root"""
    try:
        return absolute_path.relative_to(PROJECT_ROOT)
    except ValueError:
        return absolute_path
