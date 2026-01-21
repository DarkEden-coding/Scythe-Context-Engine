"""Log cleanup utility for managing log file retention.

This utility can be run manually or via cron to automatically remove old log files.
"""

import argparse
import sys
from pathlib import Path

from utils.logger import cleanup_old_logs, _get_log_dir, _get_retention_days


def format_bytes(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


def main() -> int:
    """Main entry point for log cleanup utility."""
    parser = argparse.ArgumentParser(
        description="Clean up old log files from Scythe Context Engine"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="Number of days to retain (default: from SCYTHE_LOG_RETENTION_DAYS or 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override log directory path",
    )

    args = parser.parse_args()

    # Use specified log dir or default
    if args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = _get_log_dir()

    # Get retention days
    days = args.days if args.days is not None else _get_retention_days()

    print(f"Log directory: {log_dir}")
    print(f"Retention period: {days} days")
    print(f"Dry run: {args.dry_run}\n")

    # Verify log directory exists
    if not log_dir.exists():
        print("Log directory does not exist. Nothing to clean up.", file=sys.stderr)
        return 0

    # Run cleanup
    result = cleanup_old_logs(days=days, dry_run=args.dry_run, log_dir=log_dir)

    # Print results
    deleted_files = result.get("deleted_files", 0)
    freed_bytes = result.get("freed_bytes", 0)
    status = result.get("status", "unknown")

    print(f"Results:")
    print(f"  Files deleted: {deleted_files}")
    print(f"  Space freed: {format_bytes(freed_bytes)}")
    print(f"  Status: {status}")

    if args.dry_run:
        print("\nNo files were actually deleted (dry run mode)")

    # Return appropriate exit code
    if args.dry_run or status == "success":
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
