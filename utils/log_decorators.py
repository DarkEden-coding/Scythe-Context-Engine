"""Logging decorators for automatic timing and error capture.

Provides decorators for:
- @log_timing: Automatic entry/exit logging with duration tracking
- @log_errors: Automatic exception capture and logging
"""

import copy
import functools
import time
import traceback
from typing import Any, Callable, Optional, Union

from utils.logger import log_event


def log_timing(
    event_prefix: str,
    phase: Optional[str] = None,
    component: Optional[str] = None,
) -> Callable:
    """Decorator for automatic entry/exit logging with duration tracking.

    Args:
        event_prefix: Prefix for event names (e.g., 'indexing' -> 'indexing_start', 'indexing_complete')
        phase: Optional operational phase to log
        component: Optional component name to log

    Returns:
        Decorator function.

    Example:
        @log_timing('indexing', phase='indexing', component='file_processor')
        def process_files(files):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()

            # Log start event
            log_event(
                event=f"{event_prefix}_start",
                level="INFO",
                phase=phase,
                component=component,
                message=f"Starting {event_prefix}",
            )

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                # Log completion event with duration
                duration_ms = (time.time() - start_time) * 1000
                log_event(
                    event=f"{event_prefix}_complete",
                    level="INFO",
                    phase=phase,
                    component=component,
                    message=f"Completed {event_prefix}",
                    duration_ms=duration_ms,
                )

        return wrapper

    return decorator


def log_errors(
    phase: Optional[str] = None,
    component: Optional[str] = None,
    reraise: bool = True,
    default_return: Union[Any, Callable[[], Any]] = None,
) -> Callable:
    """Decorator for automatic exception capture and logging.

    Args:
        phase: Optional operational phase to log
        component: Optional component name to log
        reraise: If True (default), re-raise the exception after logging
        default_return: Value to return when reraise is False (default: None).
                       Can be an immutable value (e.g., None, 0, "default") or
                       a factory callable that returns a fresh value (e.g., lambda: []).
                       For mutable objects, use a factory to avoid shared state.

    Returns:
        Decorator function.

    Example:
        # Using a factory for mutable defaults (recommended)
        @log_errors(phase='query', component='embedder', reraise=False, default_return=lambda: [])
        def embed_query(query):
            ...

        # Using immutable defaults
        @log_errors(phase='query', component='embedder', reraise=False, default_return=0)
        def count_results(query):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log error with full context
                log_event(
                    event=f"{func.__name__}_error",
                    level="ERROR",
                    phase=phase,
                    component=component,
                    message=f"Error in {func.__name__}: {str(e)}",
                    data={
                        "function": func.__name__,
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                    },
                    error=e,
                )

                if reraise:
                    raise
                else:
                    # Handle default_return: if callable, call it; otherwise deepcopy to avoid shared mutable state
                    if callable(default_return):
                        return default_return()
                    else:
                        return copy.deepcopy(default_return)

        return wrapper

    return decorator


def log_with_data(
    event_name: str,
    level: str = "INFO",
    phase: Optional[str] = None,
    component: Optional[str] = None,
    extract_data: Optional[Callable[[Any], dict]] = None,
) -> Callable:
    """Decorator for logging with custom data extraction.

    Args:
        event_name: Name of the event to log
        level: Log level (INFO, WARNING, ERROR, DEBUG)
        phase: Optional operational phase to log
        component: Optional component name to log
        extract_data: Optional function(result) to extract logging data from the function's return value

    Returns:
        Decorator function.

    Example:
        def extract_info(result):
            return {"result_length": len(result)}

        @log_with_data('query_processed', extract_data=extract_info)
        def process_query(query):
            return result
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = func(*args, **kwargs)

            # Extract data if function provided
            data = None
            if extract_data:
                try:
                    data = extract_data(result)
                except Exception:
                    # Silently ignore extraction errors
                    pass

            log_event(
                event=event_name,
                level=level,
                phase=phase,
                component=component,
                message=f"{event_name} in {func.__name__}",
                data=data,
            )

            return result

        return wrapper

    return decorator
