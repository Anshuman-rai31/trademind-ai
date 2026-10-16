"""TradeMind Data Processing Command-Line Interface.

Usage:
    python process_dataset.py --file data/raw/sample_aapl_daily.csv --symbol AAPL --interval 1d
    python process_dataset.py --config config/market_data_nse.json --download-all
    python process_dataset.py --config config/market_data_nse.json --refresh --symbol RELIANCE.NS
    python process_dataset.py --config config/market_data_nse.json --validate-config
    python process_dataset.py --file data/processed/reliance_ns_1d_processed.csv --split
"""
import argparse
import sys
from pathlib import Path

from .exceptions import DataIngestionError
from .market_config import ConfigValidationError, MarketDataConfig
from .pipeline import MarketDataPipeline
from .split import chronological_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trademind-data",
        description="TradeMind Market Data Ingestion, NSE Configuration, and Provenance CLI",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--file", "-f",
        type=str,
        help="Path to the raw/processed OHLCV CSV file to ingest or split.",
    )
    group.add_argument(
        "--provider", "-p",
        type=str,
        choices=["yfinance"],
        help="Public research data provider to query (e.g. yfinance).",
    )
    group.add_argument(
        "--config", "-c",
        type=str,
        help="Path to central JSON market data configuration (e.g. config/market_data_nse.json).",
    )

    parser.add_argument(
        "--symbol", "-s",
        type=str,
        help="Asset ticker symbol (e.g. 'RELIANCE.NS', 'AAPL').",
    )
    parser.add_argument(
        "--interval", "-i",
        type=str,
        default=None,
        help="Time series bar interval (e.g. '1m', '5m', '15m', '1h', '1d'). Defaults to config or '1d'.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date for provider queries (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date for provider queries (YYYY-MM-DD). Note: yfinance end date is exclusive.",
    )
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download and process all symbols specified in central configuration.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force download and merge with existing validated records (duplicate protection).",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration file parameters without downloading.",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Run chronological 70/20/10 train/validation/test split on input file.",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Override raw data directory (defaults to data/raw).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Override processed output directory (defaults to data/processed).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default=None,
        help="Override metadata provenance directory (defaults to data/metadata).",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Do not save companion JSON validation report.",
    )
    return parser


def main(args=None) -> int:
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    # 1. Config validation only mode
    if parsed_args.config and parsed_args.validate_config:
        try:
            cfg = MarketDataConfig.load_from_file(parsed_args.config)
            print(f"[CONFIG OK] Successfully validated configuration '{parsed_args.config}':")
            print(f"  Market:            {cfg.market}")
            print(f"  Data Source:       {cfg.data_source}")
            print(f"  Frequency:         {cfg.frequency}")
            print(f"  Symbols:           {', '.join(cfg.symbols)}")
            print(f"  Date Range:        {cfg.date_range.start_date} -> {cfg.date_range.end_date}")
            print(f"  Price Adjustment:  auto_adjust={cfg.price_adjustment.auto_adjust}")
            print(f"  Split Ratios:      Train={cfg.split_proportions.train}, Val={cfg.split_proportions.validation}, Test={cfg.split_proportions.test}")
            print(f"  Storage Paths:     Raw='{cfg.storage_paths.raw_dir}', Processed='{cfg.storage_paths.processed_dir}', Metadata='{cfg.storage_paths.metadata_dir}'")
            return 0
        except ConfigValidationError as exc:
            print(f"\n[CONFIG ERROR] {str(exc)}", file=sys.stderr)
            return 1

    # Load configuration if provided
    config_obj: MarketDataConfig | None = None
    if parsed_args.config:
        try:
            config_obj = MarketDataConfig.load_from_file(parsed_args.config)
        except ConfigValidationError as exc:
            print(f"\n[CONFIG ERROR] {str(exc)}", file=sys.stderr)
            return 1

    raw_dir = parsed_args.raw_dir or (config_obj.storage_paths.raw_dir if config_obj else None)
    proc_dir = parsed_args.output_dir or (config_obj.storage_paths.processed_dir if config_obj else None)
    meta_dir = parsed_args.metadata_dir or (config_obj.storage_paths.metadata_dir if config_obj else None)

    pipeline = MarketDataPipeline(
        raw_dir=raw_dir,
        processed_dir=proc_dir,
        metadata_dir=meta_dir,
        config=config_obj,
    )

    try:
        # Mode A: CSV file ingestion or splitting
        if parsed_args.file:
            if parsed_args.split:
                import pandas as pd
                from .schema import normalize_column_name
                df = pd.read_csv(parsed_args.file)
                df = df.rename(columns={c: normalize_column_name(str(c)) for c in df.columns})
                ratios = config_obj.split_proportions if config_obj else None
                splits = chronological_split(
                    df,
                    train_ratio=ratios.train if ratios else 0.70,
                    val_ratio=ratios.validation if ratios else 0.20,
                    test_ratio=ratios.test if ratios else 0.10,
                )
                print("\n=== Chronological Dataset Partitioning (Phase 3 Interface) ===")
                print(f"Source:       {parsed_args.file}")
                print(f"Total Rows:   {len(df)}")
                print(f"Training:     {splits.metadata.train_count} rows ({splits.metadata.train_start} -> {splits.metadata.train_end})")
                print(f"Validation:   {splits.metadata.val_count} rows ({splits.metadata.val_start} -> {splits.metadata.val_end})")
                print(f"Final Test:   {splits.metadata.test_count} rows ({splits.metadata.test_start} -> {splits.metadata.test_end})")
                print("================================================================")
                return 0

            print(f"--> Ingesting CSV file: {parsed_args.file}")
            interval = parsed_args.interval or (config_obj.frequency if config_obj else "1d")
            dest_path, report = pipeline.process_csv(
                source_file=parsed_args.file,
                symbol=parsed_args.symbol,
                interval=interval,
                save_report=not parsed_args.no_report,
            )
            print("\n" + report.summary())
            print(f"\n[SUCCESS] Validated dataset written to:\n  {dest_path.resolve()}")
            return 0

        # Mode B: Config batch download mode
        if parsed_args.config:
            if parsed_args.download_all:
                print(f"--> Batch downloading all {len(config_obj.symbols)} symbols from config: {', '.join(config_obj.symbols)}")
                results = pipeline.process_all_configured(config=config_obj, refresh=parsed_args.refresh)
                print(f"\n=== Batch Processing Completed ({len(results)}/{len(config_obj.symbols)} Succeeded) ===")
                for sym, (path, report, prov) in results.items():
                    print(f"  {sym:14} -> {prov.download_status:9} | {report.valid_row_count:5} rows | {path.name}")
                return 0
            elif parsed_args.symbol:
                interval = parsed_args.interval or config_obj.frequency
                start = parsed_args.start or config_obj.date_range.start_date
                end = parsed_args.end or config_obj.date_range.end_date
                print(f"--> Downloading symbol '{parsed_args.symbol}' ({start} -> {end}, interval={interval})...")
                dest_path, report, prov = pipeline.download_and_process_symbol(
                    symbol=parsed_args.symbol,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=config_obj.price_adjustment.auto_adjust,
                    refresh=parsed_args.refresh,
                    save_report=not parsed_args.no_report,
                )
                print("\n" + report.summary())
                print(f"\n[SUCCESS] Provenance: {prov.download_status} | Validation: {prov.validation_status}")
                print(f"  Dataset:    {dest_path.resolve()}")
                print(f"  Provenance: {pipeline.metadata_dir / (prov.symbol.lower().replace('.', '_') + '_' + interval + '_provenance.json')}")
                return 0
            else:
                print("Error: Specify --download-all or --symbol when using --config.", file=sys.stderr)
                return 1

        # Mode C: Direct provider query
        if parsed_args.provider:
            if not parsed_args.symbol:
                print("Error: --symbol is required when querying an external provider.", file=sys.stderr)
                return 1

            interval = parsed_args.interval or "1d"
            print(f"--> Fetching data from provider '{parsed_args.provider}' for symbol '{parsed_args.symbol}'...")
            dest_path, report, prov = pipeline.download_and_process_symbol(
                symbol=parsed_args.symbol,
                start=parsed_args.start,
                end=parsed_args.end,
                interval=interval,
                auto_adjust=True,
                refresh=parsed_args.refresh,
                save_report=not parsed_args.no_report,
            )
            print("\n" + report.summary())
            print(f"\n[SUCCESS] Validated dataset written to:\n  {dest_path.resolve()}")
            return 0

        parser.print_help()
        return 1

    except DataIngestionError as exc:
        print(f"\n[DATA INGESTION ERROR] {str(exc)}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n[UNEXPECTED ERROR] {str(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
