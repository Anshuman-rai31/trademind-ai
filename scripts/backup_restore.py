"""TradeMind SQLite Database and Artifact Backup & Restore Utility (Phase 9.5B).

Uses SQLite's official online backup API (sqlite3.Connection.backup) to safely
capture live database snapshots without corrupting active connections or WAL journals.
Also packages processed datasets, metadata, and model artifacts for disaster recovery.

Usage:
    python scripts/backup_restore.py --backup
    python scripts/backup_restore.py --backup --backup-dir C:/backups/trademind
    python scripts/backup_restore.py --restore-db C:/backups/trademind/trademind_backup_TIMESTAMP.db
    python scripts/backup_restore.py --verify
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

# Locate project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INSTANCE_DIR = PROJECT_ROOT / "instance"
DEFAULT_DB_PATH = INSTANCE_DIR / "trademind.db"
DEFAULT_BACKUP_DIR = PROJECT_ROOT / "backups"


def backup_sqlite_database(
    source_db_path: Path,
    dest_backup_path: Path,
) -> Path:
    """Perform a safe, lock-free online SQLite database backup.
    
    Uses Python's sqlite3 backup API to stream pages atomically, even while
    the development server or background workers hold open read/write locks.
    """
    if not source_db_path.exists():
        raise FileNotFoundError(f"Source database not found: {source_db_path}")

    dest_backup_path.parent.mkdir(parents=True, exist_ok=True)

    src_conn = sqlite3.connect(str(source_db_path))
    dst_conn = sqlite3.connect(str(dest_backup_path))

    try:
        # Atomic online page-by-page streaming backup
        with dst_conn:
            src_conn.backup(dst_conn, pages=100, sleep=0.01)
    finally:
        dst_conn.close()
        src_conn.close()

    return dest_backup_path


def restore_sqlite_database(
    backup_db_path: Path,
    target_db_path: Path,
) -> bool:
    """Restore SQLite database from a verified backup snapshot."""
    if not backup_db_path.exists():
        raise FileNotFoundError(f"Backup file not found: {backup_db_path}")

    # Verify backup integrity first
    verify_conn = sqlite3.connect(str(backup_db_path))
    try:
        cursor = verify_conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        row = cursor.fetchone()
        if not row or row[0] != "ok":
            raise ValueError(f"Backup database integrity check failed: {row}")
    finally:
        verify_conn.close()

    target_db_path.parent.mkdir(parents=True, exist_ok=True)
    # Stream restore
    src_conn = sqlite3.connect(str(backup_db_path))
    dst_conn = sqlite3.connect(str(target_db_path))
    try:
        with dst_conn:
            src_conn.backup(dst_conn, pages=100, sleep=0.01)
    finally:
        dst_conn.close()
        src_conn.close()

    return True


def backup_artifacts_archive(
    backup_dir: Path,
    timestamp_str: str,
) -> Dict[str, Optional[Path]]:
    """Archive processed datasets, provenance metadata, and models into compressed tarballs."""
    backup_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    # 1. Market Data (processed & metadata)
    market_data_tar = backup_dir / f"market_data_{timestamp_str}.tar.gz"
    with tarfile.open(market_data_tar, "w:gz") as tar:
        for folder in ["processed", "metadata"]:
            folder_path = PROJECT_ROOT / "data" / folder
            if folder_path.exists():
                tar.add(str(folder_path), arcname=f"data/{folder}")
    results["market_data"] = market_data_tar

    # 2. Model Artifacts
    models_dir = PROJECT_ROOT / "models"
    if models_dir.exists():
        models_tar = backup_dir / f"models_{timestamp_str}.tar.gz"
        with tarfile.open(models_tar, "w:gz") as tar:
            tar.add(str(models_dir), arcname="models")
        results["models"] = models_tar
    else:
        results["models"] = None

    return results


def run_full_backup(backup_dir: Optional[Path] = None) -> Tuple[Path, Dict[str, Optional[Path]]]:
    """Execute complete backup of database, datasets, and models."""
    target_dir = backup_dir or DEFAULT_BACKUP_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    db_backup_file = target_dir / f"trademind_db_{timestamp_str}.db"

    # SQLite backup
    backup_sqlite_database(DEFAULT_DB_PATH, db_backup_file)

    # Artifacts backup
    artifacts = backup_artifacts_archive(target_dir, timestamp_str)

    return db_backup_file, artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="TradeMind Backup and Disaster Recovery Utility")
    parser.add_argument("--backup", action="store_true", help="Perform full online backup of DB and artifacts.")
    parser.add_argument("--backup-dir", type=str, default=None, help="Destination directory for backups.")
    parser.add_argument("--restore-db", type=str, default=None, help="Path to SQLite backup file to restore.")
    parser.add_argument("--target-db", type=str, default=None, help="Target destination for DB restore.")
    parser.add_argument("--verify", action="store_true", help="Verify integrity of current SQLite database.")

    args = parser.parse_args()

    if args.verify:
        if not DEFAULT_DB_PATH.exists():
            print(f"[ERROR] Database file not found at {DEFAULT_DB_PATH}")
            return 1
        conn = sqlite3.connect(str(DEFAULT_DB_PATH))
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check")
        status = cursor.fetchone()[0]
        cursor.execute("SELECT count(*) FROM assets")
        asset_count = cursor.fetchone()[0]
        print(f"[OK] Database integrity: {status}")
        print(f"[OK] Canonical assets seeded: {asset_count}")
        conn.close()
        return 0

    if args.restore_db:
        backup_path = Path(args.restore_db)
        target_path = Path(args.target_db) if args.target_db else DEFAULT_DB_PATH
        print(f"Restoring SQLite database from {backup_path} to {target_path}...")
        restore_sqlite_database(backup_path, target_path)
        print("[SUCCESS] Database restored and integrity verified.")
        return 0

    if args.backup or not any([args.backup, args.restore_db, args.verify]):
        target_dir = Path(args.backup_dir) if args.backup_dir else DEFAULT_BACKUP_DIR
        print(f"--> Initiating TradeMind online backup to: {target_dir.resolve()}...")
        db_path, artifacts = run_full_backup(target_dir)
        print(f"[SUCCESS] Database backup saved: {db_path.name} ({db_path.stat().st_size:,} bytes)")
        if artifacts.get("market_data"):
            md_path = artifacts["market_data"]
            print(f"[SUCCESS] Market data archive:  {md_path.name} ({md_path.stat().st_size:,} bytes)")
        if artifacts.get("models"):
            m_path = artifacts["models"]
            print(f"[SUCCESS] Model registry archive: {m_path.name} ({m_path.stat().st_size:,} bytes)")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
