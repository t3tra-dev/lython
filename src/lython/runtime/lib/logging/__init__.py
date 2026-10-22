"""
Logging package for Python. Based on PEP 282 and comments thereto in
comp.lang.python.

Port of CPython 3.14's Lib/logging/__init__.py.

The caller a record is attributed to is read off the live call stack
(`_frame`, the counterpart of `sys._getframe`), as warnings' is.

Deviations from CPython:
- a message is `str(msg) % args` with the `%` done by `_printf`:
  the runtime has no `%` for a template only known when the program runs.
  The conversions are CPython's (`s r a d i u f F e E g G x X o c %` with
  flags, width and precision); a single mapping argument (`"%(a)s", {...}`)
  and `*` widths are not provided.
- `extra` is kept beside the record's fields rather than set as attributes,
  and the Formatter reads both; `record.<extra key>` is not an attribute.
- the record has no `thread` (an identifier this runtime does not have) and
  `threadName` is 'MainThread'; a format that names `thread` raises the
  "Formatting field not found" error.
- a filter is a Filter (or a subclass); CPython also takes a callable.
- `{` and `$` styles substitute `{name}`, `{name:spec}`, `{name!r}`, `$name`
  and `${name}`: what the record's fields need, not all of str.format.
- `Logger.manager` and `Logger.root` are module state read through the
  logger; `setLoggerClass`, `setLogRecordFactory`, `LoggerAdapter`,
  `captureWarnings`, the handler-name registry, `shutdown` at exit and the
  locks are not provided (a stream handler flushes after every record).
"""

import sys
import os
import io
import time
import traceback
import _frame
import linecache
from _printf import format_tuple as _percent_format, parse_conversion as _parse_conversion, convert as _convert

__all__ = ['BASIC_FORMAT', 'CRITICAL', 'DEBUG', 'ERROR',
           'FATAL', 'FileHandler', 'Filter', 'Formatter', 'Handler', 'INFO',
           'LogRecord', 'Logger', 'NOTSET', 'NullHandler',
           'StreamHandler', 'WARN', 'WARNING', 'addLevelName', 'basicConfig',
           'critical', 'debug', 'disable', 'error',
           'exception', 'fatal', 'getLevelName', 'getLogger',
           'info', 'log', 'warning', 'getLevelNamesMapping',
           'lastResort', 'raiseExceptions']

__version__ = "0.5.1.2"

_startTime = time.time_ns()

raiseExceptions = True

#---------------------------------------------------------------------------
#   Level related stuff
#---------------------------------------------------------------------------

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

_levelToName: dict[int, str] = {
    CRITICAL: 'CRITICAL',
    ERROR: 'ERROR',
    WARNING: 'WARNING',
    INFO: 'INFO',
    DEBUG: 'DEBUG',
    NOTSET: 'NOTSET',
}
_nameToLevel: dict[str, int] = {
    'CRITICAL': CRITICAL,
    'FATAL': FATAL,
    'ERROR': ERROR,
    'WARN': WARNING,
    'WARNING': WARNING,
    'INFO': INFO,
    'DEBUG': DEBUG,
    'NOTSET': NOTSET,
}


def getLevelNamesMapping() -> dict[str, int]:
    return _nameToLevel.copy()


def getLevelName(level: int) -> str:
    """
    Return the textual representation of logging level 'level'.

    If the level is one of the predefined levels (CRITICAL, ERROR, WARNING,
    INFO, DEBUG) then you get the corresponding string. If you have
    associated levels with names using addLevelName then the name you have
    associated with 'level' is returned.

    Otherwise, the string "Level %s" % level is returned.
    """
    if level in _levelToName:
        return _levelToName[level]
    return "Level %s" % level


def addLevelName(level: int, levelName: str) -> None:
    """
    Associate 'levelName' with 'level'.

    This is used when converting levels to text during message formatting.
    """
    _levelToName[level] = levelName
    _nameToLevel[levelName] = level


def _checkLevel(level: int | str) -> int:
    if isinstance(level, int):
        return level
    if level not in _nameToLevel:
        raise ValueError("Unknown level: %r" % level)
    return _nameToLevel[level]


# The file this module is compiled from: a frame in it is the logging
# machinery's own, not its caller's.
_srcfile: str = __file__


def _is_internal_filename(filename: str) -> bool:
    """Signal whether a frame is a CPython or logging module internal."""
    return (filename == _srcfile or
            ("importlib" in filename and "_bootstrap" in filename))


#---------------------------------------------------------------------------
#   The logging record
#---------------------------------------------------------------------------

_ExcInfo = tuple[type[BaseException], BaseException, object]


class LogRecord(object):
    """
    A LogRecord instance represents an event being logged.

    LogRecord instances are created every time something is logged. They
    contain all the information pertinent to the event being logged. The
    main information passed in is in msg and args, which are combined
    using str(msg) % args to create the message field of the record. The
    record also includes information such as when the record was created,
    the source line where the logging call was made, and any exception
    information to be logged.
    """
    def __init__(self, name: str, level: int, pathname: str, lineno: int,
                 msg: object, args: tuple[object, ...],
                 exc_info: BaseException | None,
                 func: str | None = None, sinfo: str | None = None) -> None:
        """
        Initialize a logging record with interesting information.
        """
        ct = time.time_ns()
        self.name = name
        self.msg = msg
        self.args = args
        self.levelname = getLevelName(level)
        self.levelno = level
        self.pathname = pathname
        self.filename = os.path.basename(pathname)
        self.module = os.path.splitext(self.filename)[0]
        self.exc_info = exc_info
        self.exc_text: str | None = None      # used to cache the traceback text
        self.stack_info = sinfo
        self.lineno = lineno
        self.funcName = func
        self.created = ct / 1e9  # ns to float seconds
        # Get the number of whole milliseconds (0-999) in the fractional part of seconds.
        # Eg: 1_677_903_920_999_998_503 ns --> 999_998_503 ns--> 999 ms
        # Convert to float by adding 0.0 for historical reasons. See gh-89047
        self.msecs = (ct % 1_000_000_000) // 1_000_000 + 0.0
        if self.msecs == 999.0 and int(self.created) != ct // 1_000_000_000:
            # ns -> sec conversion can round up, e.g:
            # 1_677_903_920_999_999_900 ns --> 1_677_903_921.0 sec
            self.msecs = 0.0

        self.relativeCreated = (ct - _startTime) / 1e6
        self.threadName = 'MainThread'
        self.processName = 'MainProcess'
        self.process = os.getpid()
        self.taskName: str | None = None
        self.message = ""
        self.asctime = ""
        self._extra: dict[str, object] = {}

    def __repr__(self) -> str:
        return '<LogRecord: %s, %s, %s, %s, "%s">'%(self.name, self.levelno,
            self.pathname, self.lineno, self.msg)

    def getMessage(self) -> str:
        """
        Return the message for this LogRecord.

        Return the message for this LogRecord after merging any user-supplied
        arguments with the message.
        """
        msg = str(self.msg)
        if self.args:
            msg = _percent_format(msg, self.args)
        return msg

    def _field(self, key: str) -> object:
        """`record.__dict__[key]`, for the fields a format can name."""
        if key == "name":
            return self.name
        if key == "msg":
            return self.msg
        if key == "args":
            return self.args
        if key == "levelname":
            return self.levelname
        if key == "levelno":
            return self.levelno
        if key == "pathname":
            return self.pathname
        if key == "filename":
            return self.filename
        if key == "module":
            return self.module
        if key == "exc_info":
            return self.exc_info
        if key == "exc_text":
            return self.exc_text
        if key == "stack_info":
            return self.stack_info
        if key == "lineno":
            return self.lineno
        if key == "funcName":
            return self.funcName
        if key == "created":
            return self.created
        if key == "msecs":
            return self.msecs
        if key == "relativeCreated":
            return self.relativeCreated
        if key == "threadName":
            return self.threadName
        if key == "processName":
            return self.processName
        if key == "process":
            return self.process
        if key == "taskName":
            return self.taskName
        if key == "message":
            return self.message
        if key == "asctime":
            return self.asctime
        if key in self._extra:
            return self._extra[key]
        raise KeyError(key)


_RECORD_FIELDS = frozenset([
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "threadName", "processName",
    "process", "taskName"])


def makeLogRecord(dict: dict[str, object]) -> LogRecord:
    """
    Make a LogRecord whose attributes are defined by the specified dictionary.

    Here the dictionary's entries are the record's `extra`.
    """
    rv = LogRecord("", 0, "", 0, "", (), None, None)
    for key, value in dict.items():
        rv._extra[key] = value
    return rv


#---------------------------------------------------------------------------
#   Formatter classes and functions
#---------------------------------------------------------------------------

def _record_value(record: LogRecord, key: str,
                  defaults: dict[str, object] | None) -> object:
    try:
        return record._field(key)
    except KeyError:
        if defaults is not None and key in defaults:
            return defaults[key]
        raise


class PercentStyle(object):

    default_format = '%(message)s'
    asctime_format = '%(asctime)s'
    asctime_search = '%(asctime)'

    def __init__(self, fmt: str | None, *,
                 defaults: dict[str, object] | None = None) -> None:
        self._fmt = fmt or self.default_format
        self._defaults = defaults

    def usesTime(self) -> bool:
        return self._fmt.find(self.asctime_search) >= 0

    def validate(self) -> None:
        """Validate the input format, ensure it matches the correct style"""
        index = 0
        found = False
        fmt = self._fmt
        while index < len(fmt):
            if fmt[index] != "%":
                index += 1
                continue
            if index + 1 < len(fmt) and fmt[index + 1] == "(":
                end = fmt.find(")", index)
                if end < 0:
                    break
                flags, width, precision, conversion, index = \
                    _parse_conversion(fmt, end + 1)
                if conversion not in "diouxXeEfFgGcrsa%":
                    break
                found = True
                continue
            index += 2
        if not found:
            raise ValueError("Invalid format '%s' for '%s' style" % (self._fmt, self.default_format[0]))

    def _format(self, record: LogRecord) -> str:
        fmt = self._fmt
        out: list[str] = []
        index = 0
        while index < len(fmt):
            char = fmt[index]
            if char != "%":
                out.append(char)
                index += 1
                continue
            if index + 1 < len(fmt) and fmt[index + 1] == "(":
                end = fmt.find(")", index)
                if end < 0:
                    raise ValueError("incomplete format key")
                key = fmt[index + 2:end]
                flags, width, precision, conversion, index = \
                    _parse_conversion(fmt, end + 1)
                if conversion == "%":
                    out.append("%")
                    continue
                value = _record_value(record, key, self._defaults)
                out.append(_convert(conversion, flags, width, precision, value))
                continue
            flags, width, precision, conversion, index = \
                _parse_conversion(fmt, index + 1)
            if conversion != "%":
                raise TypeError("format requires a mapping")
            out.append("%")
        return "".join(out)

    def format(self, record: LogRecord) -> str:
        try:
            return self._format(record)
        except KeyError as e:
            raise ValueError('Formatting field not found in record: %s' % e)


def _format_field(value: object, spec: str) -> str:
    if not spec:
        return str(value)
    if isinstance(value, bool):
        return format(str(value), spec)
    if isinstance(value, int):
        return format(value, spec)
    if isinstance(value, float):
        return format(value, spec)
    return format(str(value), spec)


class StrFormatStyle(PercentStyle):
    default_format = '{message}'
    asctime_format = '{asctime}'
    asctime_search = '{asctime'

    def _format(self, record: LogRecord) -> str:
        fmt = self._fmt
        out: list[str] = []
        index = 0
        while index < len(fmt):
            char = fmt[index]
            if char == "{" and index + 1 < len(fmt) and fmt[index + 1] == "{":
                out.append("{")
                index += 2
                continue
            if char == "}" and index + 1 < len(fmt) and fmt[index + 1] == "}":
                out.append("}")
                index += 2
                continue
            if char != "{":
                out.append(char)
                index += 1
                continue
            end = fmt.find("}", index)
            if end < 0:
                raise ValueError("expected '}' before end of string")
            field = fmt[index + 1:end]
            index = end + 1
            spec = ""
            colon = field.find(":")
            if colon >= 0:
                spec = field[colon + 1:]
                field = field[:colon]
            conversion = ""
            bang = field.find("!")
            if bang >= 0:
                conversion = field[bang + 1:]
                field = field[:bang]
            value = _record_value(record, field, self._defaults)
            if conversion == "r":
                out.append(_format_field(repr(value), spec))
            elif conversion == "a":
                out.append(_format_field(ascii(value), spec))
            else:
                out.append(_format_field(value, spec))
        return "".join(out)

    def validate(self) -> None:
        """Validate the input format, ensure it is the correct string formatting style"""
        if self._fmt.find("{") < 0:
            raise ValueError('invalid format: no fields')


class StringTemplateStyle(PercentStyle):
    default_format = '${message}'
    asctime_format = '${asctime}'
    asctime_search = '${asctime}'

    def usesTime(self) -> bool:
        fmt = self._fmt
        return fmt.find('$asctime') >= 0 or fmt.find(self.asctime_search) >= 0

    def validate(self) -> None:
        if self._fmt.find("$") < 0:
            raise ValueError('invalid format: no fields')

    def _format(self, record: LogRecord) -> str:
        fmt = self._fmt
        out: list[str] = []
        index = 0
        while index < len(fmt):
            char = fmt[index]
            if char != "$":
                out.append(char)
                index += 1
                continue
            if index + 1 < len(fmt) and fmt[index + 1] == "$":
                out.append("$")
                index += 2
                continue
            if index + 1 < len(fmt) and fmt[index + 1] == "{":
                end = fmt.find("}", index)
                if end < 0:
                    raise ValueError("Invalid placeholder in string")
                key = fmt[index + 2:end]
                index = end + 1
            else:
                end = index + 1
                while end < len(fmt) and (fmt[end].isalnum() or fmt[end] == "_"):
                    end += 1
                key = fmt[index + 1:end]
                index = end
            out.append(str(_record_value(record, key, self._defaults)))
        return "".join(out)


BASIC_FORMAT = "%(levelname)s:%(name)s:%(message)s"


def _make_style(style: str, fmt: str | None,
                defaults: dict[str, object] | None) -> PercentStyle:
    if style == '%':
        return PercentStyle(fmt, defaults=defaults)
    if style == '{':
        return StrFormatStyle(fmt, defaults=defaults)
    if style == '$':
        return StringTemplateStyle(fmt, defaults=defaults)
    raise ValueError('Style must be one of: %,{,$')


class Formatter(object):
    """
    Formatter instances are used to convert a LogRecord to text.

    Formatters need to know how a LogRecord is constructed. They are
    responsible for converting a LogRecord to (usually) a string which can
    be interpreted by either a human or an external system. The base Formatter
    allows a formatting string to be specified. If none is supplied, the
    style-dependent default value, "%(message)s", "{message}", or
    "${message}", is used.
    """

    default_time_format = '%Y-%m-%d %H:%M:%S'
    default_msec_format = '%s,%03d'

    def __init__(self, fmt: str | None = None, datefmt: str | None = None,
                 style: str = '%', validate: bool = True, *,
                 defaults: dict[str, object] | None = None) -> None:
        """
        Initialize the formatter with specified format strings.

        Initialize the formatter either with the specified format string, or a
        default as described above. Allow for specialized date formatting with
        the optional datefmt argument. If datefmt is omitted, you get an
        ISO8601-like (or RFC 3339-like) format.
        """
        self._style = _make_style(style, fmt, defaults)
        if validate:
            self._style.validate()
        self._fmt = self._style._fmt
        self.datefmt = datefmt

    def formatTime(self, record: LogRecord, datefmt: str | None = None) -> str:
        """
        Return the creation time of the specified LogRecord as formatted text.
        """
        ct = time.localtime(int(record.created))
        if datefmt:
            s = time.strftime(datefmt, ct)
        else:
            s = time.strftime(self.default_time_format, ct)
            if self.default_msec_format:
                s = _percent_format(self.default_msec_format,
                                    (s, record.msecs))
        return s

    def formatException(self, ei: BaseException) -> str:
        """
        Format and return the specified exception information as a string.

        This default implementation just uses
        traceback.format_exception().
        """
        s = "".join(traceback.format_exception(ei))
        if s[-1:] == "\n":
            s = s[:-1]
        return s

    def usesTime(self) -> bool:
        """
        Check if the format uses the creation time of the record.
        """
        return self._style.usesTime()

    def formatMessage(self, record: LogRecord) -> str:
        return self._style.format(record)

    def formatStack(self, stack_info: str) -> str:
        """
        This method is provided as an extension point for specialized
        formatting of stack information.
        """
        return stack_info

    def format(self, record: LogRecord) -> str:
        """
        Format the specified record as text.

        The record's attribute dictionary is used as the operand to a
        string formatting operation which yields the returned string.
        Before formatting the dictionary, a couple of preparatory steps
        are carried out. The message attribute of the record is computed
        using LogRecord.getMessage(). If the formatting string uses the
        time (as determined by a call to usesTime(), formatTime() is
        called to format the event time. If there is exception information,
        it is formatted using formatException() and appended to the message.
        """
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        s = self.formatMessage(record)
        exc_info = record.exc_info
        if exc_info is not None:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(exc_info)
        exc_text = record.exc_text
        if exc_text:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + exc_text
        stack_info = record.stack_info
        if stack_info:
            if s[-1:] != "\n":
                s = s + "\n"
            s = s + self.formatStack(stack_info)
        return s


_defaultFormatter = Formatter()


#---------------------------------------------------------------------------
#   Filter classes and functions
#---------------------------------------------------------------------------

class Filter(object):
    """
    Filter instances are used to perform arbitrary filtering of LogRecords.

    Loggers and Handlers can optionally use Filter instances to filter
    records as desired. The base filter class only allows events which are
    below a certain point in the logger hierarchy. For example, a filter
    initialized with "A.B" will allow events logged by loggers "A.B",
    "A.B.C", "A.B.C.D", "A.B.D" etc. but not "A.BB", "B.A.B" etc. If
    initialized with the empty string, all events are passed.
    """
    def __init__(self, name: str = '') -> None:
        """
        Initialize a filter.

        Initialize with the name of the logger which, together with its
        children, will have its events allowed through the filter. If no
        name is specified, allow every event.
        """
        self.name = name
        self.nlen = len(name)

    def filter(self, record: LogRecord) -> bool:
        """
        Determine if the specified record is to be logged.

        Returns True if the record should be logged, or False otherwise.
        """
        if self.nlen == 0:
            return True
        elif self.name == record.name:
            return True
        elif record.name.find(self.name, 0, self.nlen) != 0:
            return False
        return (record.name[self.nlen] == ".")


class Filterer(object):
    """
    A base class for loggers and handlers which allows them to share
    common code.
    """
    def __init__(self) -> None:
        """
        Initialize the list of filters to be an empty list.
        """
        self.filters: list[Filter] = []

    def addFilter(self, filter: Filter) -> None:
        """
        Add the specified filter to this handler.
        """
        if not (filter in self.filters):
            self.filters.append(filter)

    def removeFilter(self, filter: Filter) -> None:
        """
        Remove the specified filter from this handler.
        """
        if filter in self.filters:
            self.filters.remove(filter)

    def filter(self, record: LogRecord) -> bool:
        """
        Determine if a record is loggable by consulting all the filters.

        The default is to allow the record to be logged; any filter can veto
        this by returning a false value.
        """
        for f in self.filters:
            if not f.filter(record):
                return False
        return True


#---------------------------------------------------------------------------
#   Handler classes and functions
#---------------------------------------------------------------------------

class Handler(Filterer):
    """
    Handler instances dispatch logging events to specific destinations.

    The base handler class. Acts as a placeholder which defines the Handler
    interface. Handlers can optionally use Formatter instances to format
    records as desired. By default, no formatter is specified; in this case,
    the 'raw' message as determined by record.message is logged.
    """
    def __init__(self, level: int | str = NOTSET) -> None:
        """
        Initializes the instance - basically setting the formatter to None
        and the filter list to empty.
        """
        Filterer.__init__(self)
        self.name: str | None = None
        self.level = _checkLevel(level)
        self.formatter: Formatter | None = None
        self._closed = False

    def get_name(self) -> str | None:
        return self.name

    def set_name(self, name: str | None) -> None:
        self.name = name

    def acquire(self) -> None:
        """
        Acquire the I/O thread lock.
        """
        pass

    def release(self) -> None:
        """
        Release the I/O thread lock.
        """
        pass

    def setLevel(self, level: int | str) -> None:
        """
        Set the logging level of this handler.  level must be an int or a str.
        """
        self.level = _checkLevel(level)

    def format(self, record: LogRecord) -> str:
        """
        Format the specified record.

        If a formatter is set, use it. Otherwise, use the default formatter
        for the module.
        """
        fmt = self.formatter
        if fmt is None:
            return _defaultFormatter.format(record)
        return fmt.format(record)

    def emit(self, record: LogRecord) -> None:
        """
        Do whatever it takes to actually log the specified logging record.

        This version is intended to be implemented by subclasses and so
        raises a NotImplementedError.
        """
        raise NotImplementedError('emit must be implemented '
                                  'by Handler subclasses')

    def handle(self, record: LogRecord) -> bool:
        """
        Conditionally emit the specified logging record.

        Emission depends on filters which may have been added to the handler.
        Wrap the actual emission of the record with acquisition/release of
        the I/O thread lock.

        Returns an instance of the log record that was emitted
        if it passed all filters, otherwise a false value is returned.
        """
        rv = self.filter(record)
        if rv:
            self.emit(record)
        return rv

    def setFormatter(self, fmt: Formatter | None) -> None:
        """
        Set the formatter for this handler.
        """
        self.formatter = fmt

    def flush(self) -> None:
        """
        Ensure all logging output has been flushed.

        This version does nothing and is intended to be implemented by
        subclasses.
        """
        pass

    def close(self) -> None:
        """
        Tidy up any resources used by the handler.
        """
        self._closed = True

    def handleError(self, record: LogRecord, exc: BaseException) -> None:
        """
        Handle errors which occur during an emit() call.

        This method should be called from handlers when an exception is
        encountered during an emit() call. If raiseExceptions is false,
        exceptions get silently ignored.
        """
        if raiseExceptions:
            try:
                sys.stderr.write('--- Logging error ---\n')
                sys.stderr.write("".join(traceback.format_exception(exc)))
                sys.stderr.write('Logged from file %s, line %s\n' % (
                                 record.filename, record.lineno))
                sys.stderr.write('Message: %r\n'
                                 'Arguments: %s\n' % (record.msg,
                                                      record.args))
            except OSError: #pragma: no cover
                pass    # see issue 5971

    def __repr__(self) -> str:
        level = getLevelName(self.level)
        return '<%s (%s)>' % (type(self).__name__, level)


def _same_stream(a: io.TextIOWrapper | io.StringIO,
                 b: io.TextIOWrapper | io.StringIO) -> bool:
    # ⛔ Not `a is b` on the unions: identity is asked of one class at a time.
    if isinstance(a, io.StringIO) and isinstance(b, io.StringIO):
        return a is b
    if isinstance(a, io.TextIOWrapper) and isinstance(b, io.TextIOWrapper):
        return a is b
    return False


class StreamHandler(Handler):
    """
    A handler class which writes logging records, appropriately formatted,
    to a stream. Note that this class does not close the stream, as
    sys.stdout or sys.stderr may be used.
    """

    terminator = '\n'

    def __init__(self, stream: io.TextIOWrapper | io.StringIO | None = None) -> None:
        """
        Initialize the handler.

        If stream is not specified, sys.stderr is used.
        """
        Handler.__init__(self)
        if stream is None:
            stream = sys.stderr
        self.stream = stream

    def _get_stream(self) -> io.TextIOWrapper | io.StringIO:
        return self.stream

    def flush(self) -> None:
        """
        Flushes the stream.
        """
        self._get_stream().flush()

    def emit(self, record: LogRecord) -> None:
        """
        Emit a record.

        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.
        """
        try:
            msg = self.format(record)
            stream = self._get_stream()
            # issue 35046: merged two stream.writes into one.
            stream.write(msg + self.terminator)
            self.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception as e:
            self.handleError(record, e)

    def setStream(self, stream: io.TextIOWrapper | io.StringIO) -> io.TextIOWrapper | io.StringIO | None:
        """
        Sets the StreamHandler's stream to the specified value,
        if it is different.

        Returns the old stream, if the stream was changed, or None
        if it wasn't.
        """
        if _same_stream(stream, self.stream):
            return None
        result = self.stream
        self.flush()
        self.stream = stream
        return result

    def __repr__(self) -> str:
        level = getLevelName(self.level)
        return '<%s (%s)>' % (type(self).__name__, level)


class FileHandler(StreamHandler):
    """
    A handler class which writes formatted logging records to disk files.
    """
    def __init__(self, filename: str, mode: str = 'a',
                 encoding: str | None = None, delay: bool = False,
                 errors: str | None = None) -> None:
        """
        Open the specified file and use it as the stream for logging.
        """
        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.errors = errors
        self.delay = delay
        StreamHandler.__init__(self, open(self.baseFilename, self.mode))

    def close(self) -> None:
        """
        Closes the stream.
        """
        try:
            self.flush()
        finally:
            stream = self.stream
            if isinstance(stream, io.TextIOWrapper):
                stream.close()
            StreamHandler.close(self)

    def __repr__(self) -> str:
        level = getLevelName(self.level)
        return '<%s %s (%s)>' % (type(self).__name__, self.baseFilename, level)


class _StderrHandler(StreamHandler):
    """
    This class is like a StreamHandler using sys.stderr, but always uses
    whatever sys.stderr is currently set to rather than the value of
    sys.stderr at handler construction time.
    """
    def __init__(self, level: int | str = NOTSET) -> None:
        """
        Initialize the handler.
        """
        StreamHandler.__init__(self)
        self.setLevel(level)

    def _get_stream(self) -> io.TextIOWrapper | io.StringIO:
        return sys.stderr


_defaultLastResort = _StderrHandler(WARNING)
lastResort: Handler = _defaultLastResort


class NullHandler(Handler):
    """
    This handler does nothing. It's intended to be used to avoid the
    "No handlers could be found for logger XXX" one-off warning. This is
    important for library code, which may contain code to log events. If a user
    of the library does not configure logging, the one-off warning might be
    produced; to avoid this, the library developer simply needs to instantiate
    a NullHandler and add it to the top-level logger of the library module or
    package.
    """
    def handle(self, record: LogRecord) -> bool:
        """Stub."""
        return False

    def emit(self, record: LogRecord) -> None:
        """Stub."""
        pass


#---------------------------------------------------------------------------
#   Manager classes and functions
#---------------------------------------------------------------------------

class PlaceHolder(object):
    """
    PlaceHolder instances are used in the Manager logger hierarchy to take
    the place of nodes for which no loggers have been defined. This class is
    intended for internal use only and not as part of the public API.
    """
    def __init__(self, alogger: Logger) -> None:
        """
        Initialize with the specified logger being a child of this placeholder.
        """
        self.loggerMap: dict[Logger, None] = { alogger : None }

    def append(self, alogger: Logger) -> None:
        """
        Add the specified logger as a child of this placeholder.
        """
        if alogger not in self.loggerMap:
            self.loggerMap[alogger] = None


class Logger(Filterer):
    """
    Instances of the Logger class represent a single logging channel. A
    "logging channel" indicates an area of an application. Exactly how an
    "area" is defined is up to the application developer. Since an
    application can have any number of areas, logging channels are identified
    by a unique string. Application areas can be nested (e.g. an area
    of "input processing" might include sub-areas "read CSV files", "read
    XLS files" and "read Gnumeric files"). To cater for this natural nesting,
    channel names are organized into a namespace hierarchy where levels are
    separated by periods, much like the Java or Python package namespace. So
    in the instance given above, channel names might be "input" for the upper
    level, and "input.csv", "input.xls" and "input.gnu" for the sub-levels.
    There is no arbitrary limit to the depth of nesting.
    """
    def __init__(self, name: str, level: int | str = NOTSET) -> None:
        """
        Initialize the logger with a name and an optional level.
        """
        Filterer.__init__(self)
        self.name = name
        self.level = _checkLevel(level)
        self.parent: Logger | None = None
        self.propagate = True
        self.handlers: list[Handler] = []
        self.disabled = False
        self._cache: dict[int, bool] = {}

    def setLevel(self, level: int | str) -> None:
        """
        Set the logging level of this logger.  level must be an int or a str.
        """
        self.level = _checkLevel(level)
        _manager._clear_cache()

    def debug(self, msg: object, *args: object,
              exc_info: bool | BaseException | None = None,
              stack_info: bool = False, stacklevel: int = 1,
              extra: dict[str, object] | None = None) -> None:
        """
        Log 'msg % args' with severity 'DEBUG'.
        """
        if self.isEnabledFor(DEBUG):
            self._log(DEBUG, msg, args, exc_info, extra, stack_info,
                      stacklevel)

    def info(self, msg: object, *args: object,
             exc_info: bool | BaseException | None = None,
             stack_info: bool = False, stacklevel: int = 1,
             extra: dict[str, object] | None = None) -> None:
        """
        Log 'msg % args' with severity 'INFO'.
        """
        if self.isEnabledFor(INFO):
            self._log(INFO, msg, args, exc_info, extra, stack_info,
                      stacklevel)

    def warning(self, msg: object, *args: object,
                exc_info: bool | BaseException | None = None,
                stack_info: bool = False, stacklevel: int = 1,
                extra: dict[str, object] | None = None) -> None:
        """
        Log 'msg % args' with severity 'WARNING'.
        """
        if self.isEnabledFor(WARNING):
            self._log(WARNING, msg, args, exc_info, extra, stack_info,
                      stacklevel)

    def error(self, msg: object, *args: object,
              exc_info: bool | BaseException | None = None,
              stack_info: bool = False, stacklevel: int = 1,
              extra: dict[str, object] | None = None) -> None:
        """
        Log 'msg % args' with severity 'ERROR'.
        """
        if self.isEnabledFor(ERROR):
            self._log(ERROR, msg, args, exc_info, extra, stack_info,
                      stacklevel)

    def exception(self, msg: object, *args: object,
                  exc_info: bool | BaseException | None = True,
                  stack_info: bool = False, stacklevel: int = 1,
                  extra: dict[str, object] | None = None) -> None:
        """
        Convenience method for logging an ERROR with exception information.
        """
        if self.isEnabledFor(ERROR):
            self._log(ERROR, msg, args, exc_info, extra, stack_info,
                      stacklevel)

    def critical(self, msg: object, *args: object,
                 exc_info: bool | BaseException | None = None,
                 stack_info: bool = False, stacklevel: int = 1,
                 extra: dict[str, object] | None = None) -> None:
        """
        Log 'msg % args' with severity 'CRITICAL'.
        """
        if self.isEnabledFor(CRITICAL):
            self._log(CRITICAL, msg, args, exc_info, extra, stack_info,
                      stacklevel)

    def fatal(self, msg: object, *args: object,
              exc_info: bool | BaseException | None = None,
              stack_info: bool = False, stacklevel: int = 1,
              extra: dict[str, object] | None = None) -> None:
        """
        Don't use this method, use critical() instead.
        """
        if self.isEnabledFor(CRITICAL):
            self._log(CRITICAL, msg, args, exc_info, extra, stack_info,
                      stacklevel)

    def log(self, level: int, msg: object, *args: object,
            exc_info: bool | BaseException | None = None,
            stack_info: bool = False, stacklevel: int = 1,
            extra: dict[str, object] | None = None) -> None:
        """
        Log 'msg % args' with the integer severity 'level'.
        """
        if self.isEnabledFor(level):
            self._log(level, msg, args, exc_info, extra, stack_info,
                      stacklevel)

    def findCaller(self, stack_info: bool = False,
                   stacklevel: int = 1) -> tuple[str, int, str, str | None]:
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.

        Level N of `_frame` here is `sys._getframe(N)` here: the walk starts
        at this frame, as CPython's `currentframe()` does, and reads every
        frame in this body.
        """
        depth = _frame.depth()
        level = 0
        while stacklevel > 0:
            if level + 1 >= depth:
                ## If we want to use the last (deepest) frame:
                break
            level += 1
            if not _is_internal_filename(_frame.file(level)):
                stacklevel -= 1
        sinfo: str | None = None
        if stack_info:
            parts = ["Stack (most recent call last):\n"]
            outer = depth - 1
            while outer >= level:
                filename = _frame.file(outer)
                lineno = _frame.line(outer)
                parts.append('  File "%s", line %d, in %s\n'
                             % (filename, lineno, _frame.name(outer)))
                line = linecache.getline(filename, lineno).strip()
                if line:
                    parts.append("    %s\n" % line)
                outer -= 1
            sinfo = "".join(parts)
            if sinfo[-1] == '\n':
                sinfo = sinfo[:-1]
        return _frame.file(level), _frame.line(level), _frame.name(level), sinfo

    def makeRecord(self, name: str, level: int, fn: str, lno: int,
                   msg: object, args: tuple[object, ...],
                   exc_info: BaseException | None,
                   func: str | None = None,
                   extra: dict[str, object] | None = None,
                   sinfo: str | None = None) -> LogRecord:
        """
        A factory method which can be overridden in subclasses to create
        specialized LogRecords.
        """
        rv = LogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        if extra is not None:
            for key in extra:
                if (key in ["message", "asctime"]) or (key in _RECORD_FIELDS):
                    raise KeyError("Attempt to overwrite %r in LogRecord" % key)
                rv._extra[key] = extra[key]
        return rv

    def _log(self, level: int, msg: object, args: tuple[object, ...],
             exc_info: bool | BaseException | None = None,
             extra: dict[str, object] | None = None,
             stack_info: bool = False, stacklevel: int = 1) -> None:
        """
        Low-level logging routine which creates a LogRecord and then calls
        all the handlers of this logger to handle the record.
        """
        fn, lno, func, sinfo = self.findCaller(stack_info, stacklevel)
        exception: BaseException | None = None
        if isinstance(exc_info, BaseException):
            exception = exc_info
        elif exc_info is not None and exc_info:
            exception = sys.exc_info()[1]
        record = self.makeRecord(self.name, level, fn, lno, msg, args,
                                 exception, func, extra, sinfo)
        self.handle(record)

    def handle(self, record: LogRecord) -> None:
        """
        Call the handlers for the specified record.

        This method is used for unpickled records received from a socket, as
        well as those created locally. Logger-level filtering is applied.
        """
        if self.disabled:
            return
        if not self.filter(record):
            return
        self.callHandlers(record)

    def addHandler(self, hdlr: Handler) -> None:
        """
        Add the specified handler to this logger.
        """
        if not (hdlr in self.handlers):
            self.handlers.append(hdlr)

    def removeHandler(self, hdlr: Handler) -> None:
        """
        Remove the specified handler from this logger.
        """
        if hdlr in self.handlers:
            # Replace the list instead of mutating it in place, so that
            # callHandlers() can iterate it without a lock (gh-79366).
            handlers = self.handlers.copy()
            handlers.remove(hdlr)
            self.handlers = handlers

    def hasHandlers(self) -> bool:
        """
        See if this logger has any handlers configured.

        Loop through all handlers for this logger and its parents in the
        logger hierarchy. Return True if a handler was found, else False.
        Stop searching up the hierarchy whenever a logger with the "propagate"
        attribute set to zero is found - that will be the last logger which
        is checked for the existence of handlers.
        """
        c: Logger | None = self
        rv = False
        while c is not None:
            if c.handlers:
                rv = True
                break
            if not c.propagate:
                break
            else:
                c = c.parent
        return rv

    def callHandlers(self, record: LogRecord) -> None:
        """
        Pass a record to all relevant handlers.

        Loop through all handlers for this logger and its parents in the
        logger hierarchy. If no handler was found, output a one-off error
        message to sys.stderr. Stop searching up the hierarchy whenever a
        logger with the "propagate" attribute set to zero is found - that
        will be the last logger whose handlers are called.
        """
        c: Logger | None = self
        found = 0
        while c is not None:
            for hdlr in c.handlers:
                found = found + 1
                if record.levelno >= hdlr.level:
                    hdlr.handle(record)
            if not c.propagate:
                c = None    #break out
            else:
                c = c.parent
        if (found == 0):
            if record.levelno >= lastResort.level:
                lastResort.handle(record)

    def getEffectiveLevel(self) -> int:
        """
        Get the effective level for this logger.

        Loop through this logger and its parents in the logger hierarchy,
        looking for a non-zero logging level. Return the first one found.
        """
        logger: Logger | None = self
        while logger is not None:
            if logger.level != 0:
                return logger.level
            logger = logger.parent
        return NOTSET

    def isEnabledFor(self, level: int) -> bool:
        """
        Is this logger enabled for level 'level'?
        """
        if self.disabled:
            return False
        if level in self._cache:
            return self._cache[level]
        is_enabled = False
        if _manager.disable >= level:
            is_enabled = False
        else:
            is_enabled = level >= self.getEffectiveLevel()
        self._cache[level] = is_enabled
        return is_enabled

    def getChild(self, suffix: str) -> Logger:
        """
        Get a logger which is a descendant to this one.

        This is a convenience method, such that

        logging.getLogger('abc').getChild('def.ghi')

        is the same as

        logging.getLogger('abc.def.ghi')

        It's useful, for example, when the parent logger is named using
        __name__ rather than a literal string.
        """
        if root is not self:
            suffix = '.'.join((self.name, suffix))
        return _manager.getLogger(suffix)

    def __repr__(self) -> str:
        level = getLevelName(self.getEffectiveLevel())
        return '<%s %s (%s)>' % (type(self).__name__, self.name, level)


class RootLogger(Logger):
    """
    A root logger is not that different to any other logger, except that
    it must have a logging level and there is only one instance of it in
    the hierarchy.
    """
    def __init__(self, level: int | str) -> None:
        """
        Initialize the logger with the name "root".
        """
        Logger.__init__(self, "root", level)


class Manager(object):
    """
    There is [under normal circumstances] just one Manager instance, which
    holds the hierarchy of loggers.
    """
    def __init__(self, rootnode: RootLogger) -> None:
        """
        Initialize the manager with the root node of the logger hierarchy.
        """
        self.root = rootnode
        self.disable = 0
        self.emittedNoHandlerWarning = False
        self.loggerDict: dict[str, Logger | PlaceHolder] = {}

    def getLogger(self, name: str) -> Logger:
        """
        Get a logger with the specified name (channel name), creating it
        if it doesn't yet exist. This name is a dot-separated hierarchical
        name, such as "a", "a.b", "a.b.c" or similar.

        If a PlaceHolder existed for the specified name [i.e. the logger
        didn't exist but a child of it did], replace it with the created
        logger and fix up the parent/child references which pointed to the
        placeholder to now point to the logger.
        """
        if name in self.loggerDict:
            found = self.loggerDict[name]
            if isinstance(found, Logger):
                return found
            rv = Logger(name)
            self.loggerDict[name] = rv
            self._fixupChildren(found, rv)
            self._fixupParents(rv)
            return rv
        rv = Logger(name)
        self.loggerDict[name] = rv
        self._fixupParents(rv)
        return rv

    def _fixupParents(self, alogger: Logger) -> None:
        """
        Ensure that there are either loggers or placeholders all the way
        from the specified logger to the root of the logger hierarchy.
        """
        name = alogger.name
        i = name.rfind(".")
        rv: Logger | None = None
        while (i > 0) and rv is None:
            substr = name[:i]
            if substr not in self.loggerDict:
                self.loggerDict[substr] = PlaceHolder(alogger)
            else:
                obj = self.loggerDict[substr]
                if isinstance(obj, Logger):
                    rv = obj
                else:
                    obj.append(alogger)
            i = name.rfind(".", 0, i - 1)
        if rv is None:
            alogger.parent = self.root
        else:
            alogger.parent = rv

    def _fixupChildren(self, ph: PlaceHolder, alogger: Logger) -> None:
        """
        Ensure that children of the placeholder ph are connected to the
        specified logger.
        """
        name = alogger.name
        namelen = len(name)
        for c in ph.loggerMap.keys():
            #The if means ... if not c.parent.name.startswith(nm)
            parent = c.parent
            if parent is not None and parent.name[:namelen] != name:
                alogger.parent = parent
                c.parent = alogger

    def _clear_cache(self) -> None:
        """
        Clear the cache for all loggers in loggerDict
        Called when level changes are made
        """
        for logger in self.loggerDict.values():
            if isinstance(logger, Logger):
                logger._cache.clear()
        self.root._cache.clear()


root = RootLogger(WARNING)
_manager = Manager(root)


#---------------------------------------------------------------------------
# Configuration classes and functions
#---------------------------------------------------------------------------

def basicConfig(*, filename: str | None = None, filemode: str = 'a',
                format: str | None = None, datefmt: str | None = None,
                style: str = '%', level: int | str | None = None,
                stream: io.TextIOWrapper | io.StringIO | None = None,
                handlers: list[Handler] | None = None, force: bool = False,
                encoding: str | None = None,
                errors: str | None = 'backslashreplace') -> None:
    """
    Do basic configuration for the logging system.

    This function does nothing if the root logger already has handlers
    configured, unless the keyword argument *force* is set to ``True``.
    It is a convenience method intended for use by simple scripts
    to do one-shot configuration of the logging package.

    The default behaviour is to create a StreamHandler which writes to
    sys.stderr, set a formatter using the BASIC_FORMAT format string, and
    add the handler to the root logger.
    """
    if force:
        for h in root.handlers[:]:
            root.removeHandler(h)
            h.close()
    if len(root.handlers) == 0:
        if handlers is None:
            if stream is not None and filename is not None:
                raise ValueError("'stream' and 'filename' should not be "
                                 "specified together")
        else:
            if stream is not None or filename is not None:
                raise ValueError("'stream' or 'filename' should not be "
                                 "specified together with 'handlers'")
        made: list[Handler] = []
        if handlers is None:
            if filename is not None:
                made.append(FileHandler(filename, filemode,
                                        encoding=encoding, errors=errors))
            else:
                made.append(StreamHandler(stream))
        else:
            made = handlers
        fs = format if format is not None else _make_style(style, None, None).default_format
        fmt = Formatter(fs, datefmt, style)
        for h in made:
            if h.formatter is None:
                h.setFormatter(fmt)
            root.addHandler(h)
        if level is not None:
            root.setLevel(level)


#---------------------------------------------------------------------------
# Utility functions at module level.
# Basically delegate everything to the root logger.
#---------------------------------------------------------------------------

def getLogger(name: str | None = None) -> Logger:
    """
    Return a logger with the specified name, creating it if necessary.

    If no name is specified, return the root logger.
    """
    if not name or name == root.name:
        return root
    return _manager.getLogger(name)


def critical(msg: object, *args: object,
             exc_info: bool | BaseException | None = None,
             stack_info: bool = False, stacklevel: int = 1,
             extra: dict[str, object] | None = None) -> None:
    """
    Log a message with severity 'CRITICAL' on the root logger. If the logger
    has no handlers, call basicConfig() to add a console handler with a
    pre-defined format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    if root.isEnabledFor(CRITICAL):
        root._log(CRITICAL, msg, args, exc_info, extra, stack_info,
                  stacklevel)


def fatal(msg: object, *args: object,
          exc_info: bool | BaseException | None = None,
          stack_info: bool = False, stacklevel: int = 1,
          extra: dict[str, object] | None = None) -> None:
    """
    Don't use this function, use critical() instead.
    """
    if len(root.handlers) == 0:
        basicConfig()
    if root.isEnabledFor(CRITICAL):
        root._log(CRITICAL, msg, args, exc_info, extra, stack_info,
                  stacklevel)


def error(msg: object, *args: object,
          exc_info: bool | BaseException | None = None,
          stack_info: bool = False, stacklevel: int = 1,
          extra: dict[str, object] | None = None) -> None:
    """
    Log a message with severity 'ERROR' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    if root.isEnabledFor(ERROR):
        root._log(ERROR, msg, args, exc_info, extra, stack_info,
                  stacklevel)


def exception(msg: object, *args: object,
              exc_info: bool | BaseException | None = True,
              stack_info: bool = False, stacklevel: int = 1,
              extra: dict[str, object] | None = None) -> None:
    """
    Log a message with severity 'ERROR' on the root logger, with exception
    information. If the logger has no handlers, basicConfig() is called to add
    a console handler with a pre-defined format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    if root.isEnabledFor(ERROR):
        root._log(ERROR, msg, args, exc_info, extra, stack_info,
                  stacklevel)


def warning(msg: object, *args: object,
            exc_info: bool | BaseException | None = None,
            stack_info: bool = False, stacklevel: int = 1,
            extra: dict[str, object] | None = None) -> None:
    """
    Log a message with severity 'WARNING' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    if root.isEnabledFor(WARNING):
        root._log(WARNING, msg, args, exc_info, extra, stack_info,
                  stacklevel)


def info(msg: object, *args: object,
         exc_info: bool | BaseException | None = None,
         stack_info: bool = False, stacklevel: int = 1,
         extra: dict[str, object] | None = None) -> None:
    """
    Log a message with severity 'INFO' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    if root.isEnabledFor(INFO):
        root._log(INFO, msg, args, exc_info, extra, stack_info,
                  stacklevel)


def debug(msg: object, *args: object,
          exc_info: bool | BaseException | None = None,
          stack_info: bool = False, stacklevel: int = 1,
          extra: dict[str, object] | None = None) -> None:
    """
    Log a message with severity 'DEBUG' on the root logger. If the logger has
    no handlers, call basicConfig() to add a console handler with a pre-defined
    format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    if root.isEnabledFor(DEBUG):
        root._log(DEBUG, msg, args, exc_info, extra, stack_info,
                  stacklevel)


def log(level: int, msg: object, *args: object,
        exc_info: bool | BaseException | None = None,
        stack_info: bool = False, stacklevel: int = 1,
        extra: dict[str, object] | None = None) -> None:
    """
    Log 'msg % args' with the integer severity 'level' on the root logger. If
    the logger has no handlers, call basicConfig() to add a console handler
    with a pre-defined format.
    """
    if len(root.handlers) == 0:
        basicConfig()
    if root.isEnabledFor(level):
        root._log(level, msg, args, exc_info, extra, stack_info,
                  stacklevel)


def disable(level: int | str = CRITICAL) -> None:
    """
    Disable all logging calls of severity 'level' and below.
    """
    _manager.disable = _checkLevel(level)
    _manager._clear_cache()
