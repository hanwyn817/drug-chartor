#!/usr/bin/env python3
"""
Drug-Chartor: AI-powered stability trend chart generator

This tool processes pharmaceutical stability documents and generates
interactive trend charts using LLM analysis.
"""

import argparse
import sys
from pathlib import Path

from workflow import main_workflow


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate stability trend charts from pharmaceutical documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (requires OPENAI_API_KEY environment variable)
  python main.py

  # Specify custom input directory
  python main.py --input ./my_data

  # Skip document conversion (if already done)
  python main.py --skip-conversion

  # Use custom LLM settings
  python main.py --api-url https://api.openai.com/v1 --model gpt-4o

  # Specify API key via command line (not recommended, use .env instead)
  python main.py --api-key sk-xxxxx
        """
    )

    parser.add_argument(
        "--input",
        type=str,
        default="./input",
        help="Input folder path containing raw documents (default: ./input)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="./output",
        help="Output folder for converted HTML/CSV files (default: ./output)"
    )

    parser.add_argument(
        "--extracted",
        type=str,
        default="./extracted",
        help="Folder for extracted JSON data (default: ./extracted)"
    )

    parser.add_argument(
        "--charts",
        type=str,
        default="./charts",
        help="Folder for generated HTML charts (default: ./charts)"
    )

    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY environment variable)"
    )

    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="API base URL (or set OPENAI_BASE_URL environment variable)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (or set OPENAI_MODEL environment variable)"
    )

    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Skip document conversion (use if already converted)"
    )

    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()

    print("="*80)
    print("Drug-Chartor: AI-Powered Stability Trend Chart Generator")
    print("="*80)
    print()

    # Validate paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    if args.skip_conversion:
        existing_csv = list(output_path.rglob("*.csv"))
        existing_html = list(output_path.rglob("*.html"))
        if not (existing_csv or existing_html):
            if not input_path.exists():
                print(f"❌ Error: Input directory not found: {input_path}")
                print(f"   Please create the directory or specify a valid --input path")
                sys.exit(1)
    else:
        if not input_path.exists():
            print(f"❌ Error: Input directory not found: {input_path}")
            print(f"   Please create the directory or specify a valid --input path")
            sys.exit(1)

    try:
        # Run workflow
        stats = main_workflow(
            input_dir=args.input,
            output_dir=args.output,
            extracted_dir=args.extracted,
            charts_dir=args.charts,
            api_key=args.api_key,
            base_url=args.api_url,
            model=args.model,
            skip_conversion=args.skip_conversion,
        )

        # Exit with success
        if stats["charts_generated"] > 0:
            print(f"\n✓ Successfully generated {stats['charts_generated']} chart(s)")
            print(f"  Charts are saved in: {args.charts}")
            sys.exit(0)
        else:
            print(f"\n⚠ No charts generated (no valid stability data found)")
            sys.exit(0)

    except ValueError as e:
        print(f"\n❌ Configuration error: {e}")
        print("\nPlease set your OpenAI API key:")
        print("  1. Create a .env file from .env.example")
        print("  2. Set OPENAI_API_KEY in the .env file")
        print("  3. Or use the --api-key parameter")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
