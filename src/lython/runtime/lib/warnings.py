"""Python part of the warnings subsystem.

Port of CPython 3.14's Lib/_py_warnings.py (which Lib/warnings.py re-exports),
the global-context half: `sys.flags.context_aware_warnings` is off in a
build without free threading, so catch_warnings saves and restores the module
state rather than a context variable.

The place a warning is attributed to is read off the live call stack
(`_frame`, the counterpart of `sys._getframe`): the file, the line and the
module of the frame `stacklevel` calls up.

Deviations from CPython:
- the registry `warn` keeps per module is held here, keyed by the module's
  name, not as `__warningregistry__` in the module's globals, which a
  compiled module does not have; `_clear_registries` empties them, which is
  what assigning `{}` to each `__warningregistry__` does in CPython.
- `showwarning` and `formatwarning` cannot be replaced by assigning the
  module attribute, and catch_warnings records by switching the module's own
  log rather than by replacing `_showwarnmsg_impl`.
- catch_warnings restores the filters list IN PLACE: CPython rebinds
  `warnings.filters` to a copy on entry and back on exit, so a reference to
  the list taken inside the block sees the outer list again here.
- `source` is accepted and only remembered as given or not: there is no
  tracemalloc to ask where it was allocated, so the formatted warning
  carries CPython's suggestion line, as CPython's does without tracemalloc.
- `deprecated`, `-W` options (`sys.warnoptions`) and the `module` argument
  of catch_warnings are not provided.
"""

import sys
import re
import io
import _frame
import linecache


__all__ = ["warn", "warn_explicit", "showwarning",
           "formatwarning", "filterwarnings", "simplefilter",
           "resetwarnings", "catch_warnings"]


# filters contains a sequence of filter 5-tuples
# The components of the 5-tuple are:
# - an action: error, ignore, always, all, default, module, or once
# - a compiled regex that must match the warning message
# - a class representing the warning category
# - a compiled regex that must match the module that is being warned
# - a line number for the line being warning, or 0 to mean any line
# If either if the compiled regexs are None, match anything.
filters: list[tuple[str, re.Pattern | None, type[Warning], re.Pattern | None, int]] = []


defaultaction = "default"
onceregistry: dict[tuple[str, type[Warning]], int] = {}


class _State:
    """The module state catch_warnings swaps: the filters' version and the
    log a recording block appends to."""

    def __init__(self) -> None:
        self.filters_version = 1
        self.log: list[WarningMessage] | None = None


_state = _State()

# The registry of each module that has warned, by the module's name.
_registries: dict[str, dict[tuple[str, type[Warning], int], int]] = {}
_registry_versions: dict[str, int] = {}


def _clear_registries() -> None:
    for registry in _registries.values():
        registry.clear()


def _filters_mutated_lock_held() -> None:
    _state.filters_version += 1


class WarningMessage(object):

    _WARNING_DETAILS = ("message", "category", "filename", "lineno", "file",
                        "line", "source")

    def __init__(self, message: Warning, category: type[Warning],
                 filename: str, lineno: int,
                 file: io.TextIOWrapper | io.StringIO | None = None,
                 line: str | None = None, source: bool = False) -> None:
        self.message = message
        self.category = category
        self.filename = filename
        self.lineno = lineno
        self.file = file
        self.line = line
        self.source = source
        self._category_name = category.__name__

    def __str__(self) -> str:
        return ("{message : %r, category : %r, filename : %r, lineno : %s, "
                    "line : %r}" % (self.message, self._category_name,
                                    self.filename, self.lineno, self.line))

    def __repr__(self) -> str:
        return f'<{type(self).__qualname__} {self}>'


def showwarning(message: Warning, category: type[Warning], filename: str,
                lineno: int,
                file: io.TextIOWrapper | io.StringIO | None = None,
                line: str | None = None) -> None:
    """Hook to write a warning to a file; replace if you like."""
    msg = WarningMessage(message, category, filename, lineno, file, line)
    _showwarnmsg_impl(msg)


def formatwarning(message: Warning, category: type[Warning], filename: str,
                  lineno: int, line: str | None = None) -> str:
    """Function to format a warning the standard way."""
    msg = WarningMessage(message, category, filename, lineno, None, line)
    return _formatwarnmsg_impl(msg)


def _showwarnmsg_impl(msg: WarningMessage) -> None:
    log = _state.log
    if log is not None:
        log.append(msg)
        return
    text = _formatwarnmsg_impl(msg)
    file = msg.file
    try:
        if file is None:
            sys.stderr.write(text)
        else:
            file.write(text)
    except OSError:
        # the file (probably stderr) is invalid - this warning gets lost.
        pass


def _formatwarnmsg_impl(msg: WarningMessage) -> str:
    category = msg.category.__name__
    s =  f"{msg.filename}:{msg.lineno}: {category}: {msg.message}\n"

    line = msg.line
    if line is None:
        line = linecache.getline(msg.filename, msg.lineno)
    if line:
        line = line.strip()
        s += "  %s\n" % line

    if msg.source:
        s += (f'{category}: Enable tracemalloc to get the object '
              f'allocation traceback\n')
    return s


def _check_action(action: str) -> None:
    if action not in {"error", "ignore", "always", "all", "default", "module", "once"}:
        raise ValueError(f"invalid action: {action!r}")


def filterwarnings(action: str, message: str = "",
                   category: type[Warning] = Warning, module: str = "",
                   lineno: int = 0, append: bool = False) -> None:
    """Insert an entry into the list of warnings filters (at the front).

    'action' -- one of "error", "ignore", "always", "all", "default", "module",
                or "once"
    'message' -- a regex that the warning message must match
    'category' -- a class that the warning must be a subclass of
    'module' -- a regex that the module name must match
    'lineno' -- an integer line number, 0 matches all warnings
    'append' -- if true, append to the list of filters
    """
    _check_action(action)
    if lineno < 0:
        raise ValueError("lineno must be an int >= 0")

    compiled_message: re.Pattern | None = None
    if message:
        compiled_message = re.compile(message, re.I)
    compiled_module: re.Pattern | None = None
    if module:
        compiled_module = re.compile(module)

    _add_filter((action, compiled_message, category, compiled_module, lineno),
                append)


def simplefilter(action: str, category: type[Warning] = Warning,
                 lineno: int = 0, append: bool = False) -> None:
    """Insert a simple entry into the list of warnings filters (at the front).

    A simple filter matches all modules and messages.
    'action' -- one of "error", "ignore", "always", "all", "default", "module",
                or "once"
    'category' -- a class that the warning must be a subclass of
    'lineno' -- an integer line number, 0 matches all warnings
    'append' -- if true, append to the list of filters
    """
    _check_action(action)
    if lineno < 0:
        raise ValueError("lineno must be an int >= 0")
    _add_filter((action, None, category, None, lineno), append)


def _add_filter(item: tuple[str, re.Pattern | None, type[Warning],
                            re.Pattern | None, int],
                append: bool) -> None:
    if not append:
        # Remove possible duplicate filters, so new one will be placed
        # in correct place. If append=True and duplicate exists, do nothing.
        try:
            filters.remove(item)
        except ValueError:
            pass
        filters.insert(0, item)
    else:
        if item not in filters:
            filters.append(item)
    _filters_mutated_lock_held()


def resetwarnings() -> None:
    """Clear the list of warning filters, so that no filters are active."""
    del filters[:]
    _filters_mutated_lock_held()


def _is_internal_filename(filename: str) -> bool:
    return 'importlib' in filename and '_bootstrap' in filename


def _is_filename_to_skip(filename: str,
                         skip_file_prefixes: tuple[str, ...]) -> bool:
    return any(filename.startswith(prefix) for prefix in skip_file_prefixes)


def warn(message: str | Warning, category: type[Warning] | None = None,
         stacklevel: int = 1, source: object = None, *,
         skip_file_prefixes: tuple[str, ...] = ()) -> None:
    """Issue a warning, or maybe ignore it or raise an exception."""
    # Check if message is already a Warning object
    if isinstance(message, Warning):
        category = type(message)
    # Check category argument
    if category is None:
        category = UserWarning
    if skip_file_prefixes:
        stacklevel = max(2, stacklevel)
    # Get context information. CPython runs the C `warn` (Python/_warnings.c,
    # setup_context), which has no frame of its own: stacklevel 1 is the
    # caller and a stacklevel below 1 is too. Level N of `_frame` here is
    # `sys._getframe(N)` here, so the frames are read in this body.
    depth = _frame.depth()
    level = 1
    if level < depth and (stacklevel <= 0 or
                          _is_internal_filename(_frame.file(level))):
        for x in range(stacklevel - 1):
            level += 1
    else:
        for x in range(stacklevel - 1):
            level += 1
            while level < depth and (
                    _is_internal_filename(_frame.file(level)) or
                    _is_filename_to_skip(_frame.file(level),
                                         skip_file_prefixes)):
                level += 1
            if level >= depth:
                break
    if level >= depth:
        level = -1
    filename = "<sys>"
    lineno = 0
    module = "sys"
    if level >= 0:
        filename = _frame.file(level)
        lineno = _frame.line(level)
        module = _frame.module(level)
        if not module:
            module = "<string>"
    warn_explicit(message, category, filename, lineno, module,
                  source=source is not None)


def _registry_for(module: str) -> dict[tuple[str, type[Warning], int], int]:
    if module not in _registries:
        _registries[module] = {}
        _registry_versions[module] = 0
    registry = _registries[module]
    if _registry_versions[module] != _state.filters_version:
        registry.clear()
        _registry_versions[module] = _state.filters_version
    return registry


def warn_explicit(message: str | Warning, category: type[Warning],
                  filename: str, lineno: int, module: str | None = None,
                  source: bool = False) -> None:
    if module is None:
        module = filename or "<unknown>"
        if module[-3:].lower() == ".py":
            module = module[:-3] # XXX What about leading pathname?
    text = ""
    warning: Warning
    if isinstance(message, Warning):
        text = str(message)
        category = type(message)
        warning = message
    else:
        text = message
        warning = category(message)
    key = (text, category, lineno)
    registry = _registry_for(module)
    # Quick test for common case
    if registry.get(key, 0) != 0:
        return
    # Search the filters
    action = defaultaction
    for item in filters:
        item_action, pattern, cat, mod, ln = item
        if ((pattern is None or pattern.match(text) is not None) and
            issubclass(category, cat) and
            (mod is None or mod.match(module) is not None) and
            (ln == 0 or lineno == ln)):
            action = item_action
            break
    # Early exit actions
    if action == "ignore":
        return

    if action == "error":
        raise warning
    # Other actions
    if action == "once":
        registry[key] = 1
        oncekey = (text, category)
        if onceregistry.get(oncekey, 0) != 0:
            return
        onceregistry[oncekey] = 1
    elif action in {"always", "all"}:
        pass
    elif action == "module":
        registry[key] = 1
        altkey = (text, category, 0)
        if registry.get(altkey, 0) != 0:
            return
        registry[altkey] = 1
    elif action == "default":
        registry[key] = 1

    # Prime the linecache for formatting, in case the
    # "file" is actually in a zipfile or something.
    linecache.getlines(filename)

    # Print message and context
    msg = WarningMessage(warning, category, filename, lineno, source=source)
    _showwarnmsg_impl(msg)


class catch_warnings(object):

    """A context manager that copies and restores the warnings filter upon
    exiting the context.

    The 'record' argument specifies whether warnings should be captured by a
    custom implementation of warnings.showwarning() and be appended to a list
    returned by the context manager. Otherwise None is returned by the context
    manager. The objects appended to the list are arguments whose attributes
    mirror the arguments to showwarning().

    If the 'action' argument is not None, the remaining arguments are passed
    to warnings.simplefilter() as if it were called immediately on entering the
    context.
    """

    def __init__(self, *, record: bool = False, action: str | None = None,
                 category: type[Warning] = Warning, lineno: int = 0,
                 append: bool = False) -> None:
        """Specify whether to record warnings."""
        self._record = record
        self._entered = False
        self._action = action
        self._category = category
        self._lineno = lineno
        self._append = append
        self._filters: list[tuple[str, re.Pattern | None, type[Warning],
                                  re.Pattern | None, int]] = []
        self._log: list[WarningMessage] | None = None

    def __repr__(self) -> str:
        args: list[str] = []
        if self._record:
            args.append("record=True")
        name = type(self).__name__
        return "%s(%s)" % (name, ", ".join(args))

    def _enter_recording(self) -> list[WarningMessage]:
        """`__enter__` of a recording block, for a caller that knows it is one.

        ⛔ Not `__enter__`'s `list | None` narrowed at the caller: a list of
        WarningMessage (whose `file` is a three-way union) narrowed from an
        Optional and stored in a field is refused by the ownership check
        (docs/unittest-port.md, open defects)."""
        self._record = True
        log = self.__enter__()
        if log is None:
            return []
        return log

    def __enter__(self) -> list[WarningMessage] | None:
        if self._entered:
            raise RuntimeError("Cannot enter %r twice" % self)
        self._entered = True
        self._filters = filters[:]
        self._log = _state.log
        _filters_mutated_lock_held()
        log: list[WarningMessage] | None = None
        if self._record:
            log = []
        _state.log = log
        action = self._action
        if action is not None:
            simplefilter(action, self._category, self._lineno, self._append)
        return log

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 traceback: object) -> None:
        if not self._entered:
            raise RuntimeError("Cannot exit %r without entering first" % self)
        filters[:] = self._filters
        _state.log = self._log
        _filters_mutated_lock_held()


def _setup_defaults() -> bool:
    # Several warning categories are ignored by default in regular builds
    filterwarnings("default", category=DeprecationWarning, module="__main__", append=True)
    simplefilter("ignore", category=DeprecationWarning, append=True)
    simplefilter("ignore", category=PendingDeprecationWarning, append=True)
    simplefilter("ignore", category=ImportWarning, append=True)
    simplefilter("ignore", category=ResourceWarning, append=True)
    return True


# ⛔ Not a bare `_setup_defaults()` statement: an imported module's body runs
# only its bindings, so the call is the value of one.
_defaults_installed: bool = _setup_defaults()
