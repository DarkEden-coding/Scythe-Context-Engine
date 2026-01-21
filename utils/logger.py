"""Comprehensive logging system for MCP queries in Scythe Context Engine.

Features:
- JSON-formatted logs with structured events
- Automatic timestamp and query_id injection
- Date-organized log file structure
- Context propagation using contextvars
- Automatic log cleanup for old logs
- System-level event logging
"""

import json
import logging
import os
import sys
import threading
from contextvars import ContextVar
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

# Context variable for query_id propagation
_query_id_context: ContextVar[Optional[str]] = ContextVar("query_id", default=None)
_query_context_data: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "query_context_data", default=None
)

# Global logger instance
_system_logger: Optional[logging.Logger] = None

# Cache for query-specific loggers and their handlers for cleanup
_query_loggers: Dict[str, logging.Logger] = {}
# Lock for thread-safe access to _query_loggers
_query_loggers_lock = threading.Lock()


def _get_log_dir() -> Path:
    """Get the log directory path from environment or use default."""
    log_dir = os.environ.get("SCYTHE_LOG_DIR", "./logs")
    return Path(log_dir).expanduser().absolute()


def _get_log_level() -> str:
    """Get the log level from environment or use default."""
    return os.environ.get("SCYTHE_LOG_LEVEL", "INFO").upper()


def _get_retention_days() -> int:
    """Get the log retention days from environment or use default."""
    try:
        return int(os.environ.get("SCYTHE_LOG_RETENTION_DAYS", "30"))
    except ValueError:
        return 30


class JSONLogFormatter(logging.Formatter):
    """Custom formatter that outputs JSON-formatted log entries."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted log entry.
        """
        # Get query_id from context
        query_id = _query_id_context.get()

        # Build log entry
        log_entry = {
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
        }

        # Add query_id if available
        if query_id:
            log_entry["query_id"] = query_id

        # Add message and custom fields from the record
        if hasattr(record, "event"):
            log_entry["event"] = record.event
        if hasattr(record, "phase"):
            log_entry["phase"] = record.phase
        if hasattr(record, "component"):
            log_entry["component"] = record.component
        if hasattr(record, "message") and record.message != record.getMessage():
            log_entry["message"] = record.message
        else:
            log_entry["message"] = record.getMessage()
        if hasattr(record, "data"):
            log_entry["data"] = record.data
        if hasattr(record, "duration_ms"):
            log_entry["duration_ms"] = record.duration_ms
        if hasattr(record, "error_context"):
            log_entry["error"] = record.error_context

        # Add exception info if present (merge with existing error_context)
        if record.exc_info:
            exc_error = {
                "type": record.exc_info[0].__name__
                if record.exc_info[0]
                else "Unknown",
                "message": str(record.exc_info[1]) if record.exc_info[1] else "",
                "traceback": self.formatException(record.exc_info)
                if record.exc_info
                else None,
            }
            # Merge with existing error context if present
            if "error" in log_entry:
                log_entry["error"]["exception"] = exc_error
            else:
                log_entry["error"] = exc_error

        return json.dumps(log_entry)


def init_logging_system() -> None:
    """Initialize the logging system with system-level logger.

    Creates the logs directory structure and sets up the system logger
    for capturing system-level events.
    """
    global _system_logger

    log_dir = _get_log_dir()
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create system logger
    _system_logger = logging.getLogger("scythe.system")
    _system_logger.setLevel(getattr(logging, _get_log_level()))

    # Remove and close existing handlers to prevent resource leaks
    for handler in list(_system_logger.handlers):
        _system_logger.removeHandler(handler)
        handler.close()

    # Create system.jsonl handler
    system_log_path = log_dir / "system.jsonl"
    system_handler = logging.FileHandler(system_log_path, encoding="utf-8")
    system_handler.setFormatter(JSONLogFormatter())
    system_handler.setLevel(getattr(logging, _get_log_level()))
    _system_logger.addHandler(system_handler)

    # Also add stderr handler for warnings and errors
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    _system_logger.addHandler(stderr_handler)


def create_query_logger(query_id: str) -> str:
    """Create a query-specific logger and return the log file path.

    Args:
        query_id: Unique identifier for the query.

    Returns:
        Path to the query log file.
    """
    log_dir = _get_log_dir()
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    query_log_dir = log_dir / "queries" / date_str
    query_log_dir.mkdir(parents=True, exist_ok=True)

    log_path = query_log_dir / f"q_{query_id}.jsonl"
    return str(log_path)


def set_query_context(query_id: str, **kwargs: Any) -> None:
    """Set the query context for the current async context.

    Args:
        query_id: Unique identifier for the query.
        **kwargs: Additional context data to store.
    """
    _query_id_context.set(query_id)
    context_data = dict(kwargs) if kwargs else {}
    _query_context_data.set(context_data)


def get_query_context() -> Dict[str, Any]:
    """Get the current query context.

    Returns:
        Dictionary containing query context data.
    """
    context = _query_context_data.get()
    return dict(context) if context else {}


def get_query_id() -> Optional[str]:
    """Get the current query_id from context.

    Returns:
        The query_id if set, None otherwise.
    """
    return _query_id_context.get()


def close_query_logger(query_id: str) -> None:
    """Close and cleanup a query-specific logger.

    Args:
        query_id: The query_id of the logger to close.
    """
    with _query_loggers_lock:
        if query_id in _query_loggers:
            logger = _query_loggers.pop(query_id)
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.flush()
                handler.close()


def log_event(
    event: str,
    level: str = "INFO",
    phase: Optional[str] = None,
    component: Optional[str] = None,
    message: Optional[str] = None,
    data: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[float] = None,
    error: Optional[Exception] = None,
) -> None:
    """Log a structured event.

    Args:
        event: Event name (e.g., 'query_start', 'indexing_complete').
        level: Log level (INFO, WARNING, ERROR, DEBUG, CRITICAL).
        phase: Operational phase (server, indexing, query, reranking, refinement).
        component: Component name (file_processor, embedder, etc.).
        message: Human-readable message.
        data: Event-specific data dictionary.
        duration_ms: Duration in milliseconds.
        error: Exception object if logging an error.
    """
    query_id = get_query_id()

    # Create a logger for this query or system
    if query_id:
        # Use full query_id to avoid collisions
        logger_name = f"scythe.query.{query_id}"

        # Use cached logger if available, otherwise create new one
        # Double-checked locking pattern for thread-safe logger creation
        if query_id not in _query_loggers:
            with _query_loggers_lock:
                # Re-check after acquiring lock
                if query_id not in _query_loggers:
                    logger = logging.getLogger(logger_name)
                    # Only set up handlers if this is the first time we're using this logger
                    if not logger.handlers:
                        log_path = create_query_logger(query_id)
                        handler = logging.FileHandler(log_path, encoding="utf-8")
                        handler.setFormatter(JSONLogFormatter())
                        handler.setLevel(
                            logging.DEBUG
                        )  # Capture all levels at handler level
                        logger.addHandler(handler)
                        logger.setLevel(logging.DEBUG)
                    _query_loggers[query_id] = logger
                else:
                    logger = _query_loggers[query_id]
        else:
            with _query_loggers_lock:
                logger = _query_loggers[query_id]
    else:
        # Fall back to system logger
        global _system_logger
        if _system_logger is None:
            init_logging_system()
        logger = _system_logger

    # Create log record with custom attributes
    log_method = getattr(logger, level.lower(), logger.info)

    # Prepare the message
    log_message = message or event

    # Create a custom LogRecord to attach extra data
    record = logger.makeRecord(
        name=logger.name,
        level=getattr(logging, level.upper(), logging.INFO),
        fn="<log_event>",
        lno=0,
        msg=log_message,
        args=(),
        exc_info=None,
    )

    # Attach custom fields to the record
    record.event = event
    if phase:
        record.phase = phase
    if component:
        record.component = component
    if message:
        record.message = message
    if data:
        record.data = data
    if duration_ms is not None:
        record.duration_ms = duration_ms

    # Add error context if error is provided
    if error:
        record.error_context = {
            "type": type(error).__name__,
            "message": str(error),
        }

    logger.handle(record)


def cleanup_old_logs(
    days: Optional[int] = None, dry_run: bool = False, log_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Clean up log files older than N days.

    Args:
        days: Number of days to retain. If None, uses environment variable or default.
        dry_run: If True, don't delete, just report what would be deleted.
        log_dir: Optional log directory path. If None, uses environment variable or default.

    Returns:
        Dictionary with cleanup statistics.
    """
    if days is None:
        days = _get_retention_days()

    if log_dir is None:
        log_dir = _get_log_dir()
    else:
        log_dir = Path(log_dir) if not isinstance(log_dir, Path) else log_dir
    queries_dir = log_dir / "queries"

    if not queries_dir.exists():
        return {"deleted_files": 0, "freed_bytes": 0, "status": "no_logs_directory"}

    cutoff_date = datetime.utcnow() - timedelta(days=days)
    deleted_count = 0
    freed_bytes = 0

    # Iterate through date directories
    for date_dir in queries_dir.iterdir():
        if not date_dir.is_dir():
            continue

        try:
            dir_date = datetime.strptime(date_dir.name, "%Y-%m-%d")
        except ValueError:
            # Skip directories that don't match date format
            continue

        if dir_date < cutoff_date:
            # Delete this entire directory
            for log_file in date_dir.glob("*.jsonl"):
                freed_bytes += log_file.stat().st_size
                if not dry_run:
                    log_file.unlink()
                deleted_count += 1

            if not dry_run:
                # Remove empty directory
                try:
                    date_dir.rmdir()
                except OSError:
                    # Directory not empty or other error
                    pass

    return {
        "deleted_files": deleted_count,
        "freed_bytes": freed_bytes,
        "status": "success",
    }


# Initialize logging system on module import
try:
    init_logging_system()
except Exception as e:
    print(f"Warning: Failed to initialize logging system: {e}", file=sys.stderr)
