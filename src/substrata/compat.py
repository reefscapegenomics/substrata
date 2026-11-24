"""Backward compatibility for renamed names (see the naming convention in the docs).

Renamed functions, classes and methods keep their old name as a plain alias
(``old_name = new_name  # deprecated alias (renamed <date>)``), so they are the
same object. Renamed keyword parameters of public functions keep working through
:func:`renamed_kwargs`, which maps the old keyword to the new one and warns
(:class:`DeprecationWarning`) once per function and keyword.
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Callable, Dict, Set, Tuple

__all__ = ["renamed_kwargs", "warn_renamed_kwarg"]

# (function qualname, old keyword) pairs that have already warned
_WARNED: Set[Tuple[str, str]] = set()


def _warn_once(func: Callable, old: str, new: str, stacklevel: int = 3) -> None:
    key = (f"{func.__module__}.{func.__qualname__}", old)
    if key in _WARNED:
        return
    _WARNED.add(key)
    warnings.warn(
        f"{func.__qualname__}(): the keyword {old!r} is deprecated, use {new!r}",
        DeprecationWarning,
        stacklevel=stacklevel,
    )


def warn_renamed_kwarg(func: Callable, old: str, new: str) -> None:
    """Warn (once) that ``func`` was called with the deprecated keyword ``old``.

    For renames :func:`renamed_kwargs` cannot express (e.g. an inverted flag);
    call it from ``func`` itself.
    """
    _warn_once(func, old, new, stacklevel=4)


def reset_warnings() -> None:
    """Forget which renamed keywords have warned (for tests)."""
    _WARNED.clear()


def renamed_kwargs(**renames: str) -> Callable[[Callable], Callable]:
    """Accept old keyword names of a function: ``@renamed_kwargs(old="new")``.

    A call with ``old=`` is passed on as ``new=`` (with a DeprecationWarning, once
    per function and keyword); passing both is a TypeError. The wrapper keeps the
    function's name, docstring and signature (``functools.wraps``), so
    ``__name__``-based dispatch and ``inspect.signature`` see the new function.
    """

    def decorate(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for old, new in renames.items():
                if old in kwargs:
                    if new in kwargs:
                        raise TypeError(
                            f"{func.__qualname__}() got both {old!r} (deprecated) and {new!r}"
                        )
                    _warn_once(func, old, new)
                    kwargs[new] = kwargs.pop(old)
            return func(*args, **kwargs)

        wrapper.__renamed_kwargs__: Dict[str, str] = dict(renames)  # type: ignore[attr-defined]
        return wrapper

    return decorate
