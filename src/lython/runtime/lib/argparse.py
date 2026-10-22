"""Command-line parsing library

Port of CPython 3.14's Lib/argparse.py.

The parsed values live in a Namespace that answers `args.name` through
`__getattr__`: the names a parser defines are only known when the program
runs, so every value read off a Namespace is an `object`, narrowed where it
is used (`isinstance(args.count, int)`, `str(args.name)`).

Deviations from CPython:
- help is never colored (CPython colors it only on a terminal), and the
  width is `COLUMNS` or 80, less 2: the terminal is not asked for its size.
- `type=` is a `Callable[[str], object]`; `choices=` any iterable, kept as a
  list. A Namespace has no `__dict__`: `vars(args)` is `args._get_kwargs()`.
- `action=` is a registered name or an Action subclass; a subclass is built
  with every keyword argument add_argument takes, so its `__init__` takes
  `**kwargs` (as CPython's documented example does). `register()` takes
  action classes only. An action stores into the Namespace with
  `namespace._set(dest, value)` where CPython's uses `setattr`.
- `ArgumentError.argument_name` is '' rather than None when no argument is
  named.
- `fromfile_prefix_chars`, `parents`, `FileType`, intermixed parsing,
  `convert_arg_line_to_args` and the `%`-style `usage` with other keys than
  `prog` are not provided; `help=None` passed to `add_parser` is the same as
  no help.
"""

__version__ = '1.1'
__all__ = [
    'ArgumentParser',
    'ArgumentError',
    'ArgumentTypeError',
    'BooleanOptionalAction',
    'HelpFormatter',
    'ArgumentDefaultsHelpFormatter',
    'RawDescriptionHelpFormatter',
    'RawTextHelpFormatter',
    'MetavarTypeHelpFormatter',
    'Namespace',
    'Action',
    'ONE_OR_MORE',
    'OPTIONAL',
    'PARSER',
    'REMAINDER',
    'SUPPRESS',
    'ZERO_OR_MORE',
]


import io
import os as _os
import re as _re
import sys as _sys
import textwrap
import difflib
from collections.abc import Callable, Iterable

from _printf import format_mapping as _format_mapping

SUPPRESS = '==SUPPRESS=='

OPTIONAL = '?'
ZERO_OR_MORE = '*'
ONE_OR_MORE = '+'
PARSER = 'A...'
REMAINDER = '...'
_UNRECOGNIZED_ARGS_ATTR = '_unrecognized_args'


def _is_suppress(value: object) -> bool:
    # ⛔ Not `value is SUPPRESS`: identity of a str has no meaning here, and
    # SUPPRESS is the only value of its spelling a program passes.
    return isinstance(value, str) and value == SUPPRESS


def _identity(value: str) -> object:
    return value


def _terminal_width() -> int:
    columns = _os.getenv('COLUMNS')
    if columns is not None and columns.isdigit() and int(columns) > 0:
        return int(columns)
    return 80


# ===============
# Formatting Help
# ===============

class _Section(object):

    def __init__(self, formatter: HelpFormatter, parent: _Section | None,
                 heading: str | None = None) -> None:
        self.formatter = formatter
        self.parent = parent
        self.heading = heading
        self.items: list[Callable[[], str]] = []

    def format_help(self) -> str:
        # format the indented section
        if self.parent is not None:
            self.formatter._indent()
        join = self.formatter._join_parts
        item_help = join([func() for func in self.items])
        if self.parent is not None:
            self.formatter._dedent()

        # return nothing if the section was empty
        if not item_help:
            return ''

        # add the heading if the section was non-empty
        heading_text = self.heading
        if heading_text is not None and not _is_suppress(heading_text):
            current_indent = self.formatter._current_indent
            heading = ' ' * current_indent + heading_text + ':\n'
        else:
            heading = ''

        # join the section-initial newline, the heading and the help
        return join(['\n', heading, item_help, '\n'])


class HelpFormatter(object):
    """Formatter for generating usage messages and argument help strings.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def __init__(
        self,
        prog: str,
        indent_increment: int = 2,
        max_help_position: int = 24,
        width: int | None = None,
        color: bool = True,
    ) -> None:
        # default setting for width
        if width is None:
            width = _terminal_width()
            width -= 2

        self._prog = prog
        self._indent_increment = indent_increment
        self._max_help_position = min(max_help_position,
                                      max(width - 20, indent_increment * 2))
        self._width = width

        self._current_indent = 0
        self._level = 0
        self._action_max_length = 0

        self._root_section = _Section(self, None)
        self._current_section = self._root_section

        self._whitespace_matcher = _re.compile(r'\s+', _re.ASCII)
        self._long_break_matcher = _re.compile(r'\n\n\n+')

    def _set_color(self, color: bool) -> None:
        pass

    # ===============================
    # Section and indentation methods
    # ===============================

    def _indent(self) -> None:
        self._current_indent += self._indent_increment
        self._level += 1

    def _dedent(self) -> None:
        self._current_indent -= self._indent_increment
        assert self._current_indent >= 0, 'Indent decreased below 0.'
        self._level -= 1

    def _add_item(self, func: Callable[[], str]) -> None:
        self._current_section.items.append(func)

    # ========================
    # Message building methods
    # ========================

    def start_section(self, heading: str | None) -> None:
        self._indent()
        section = _Section(self, self._current_section, heading)
        self._add_item(section.format_help)
        self._current_section = section

    def end_section(self) -> None:
        parent = self._current_section.parent
        if parent is not None:
            self._current_section = parent
        self._dedent()

    def add_text(self, text: str | None) -> None:
        if text is not None and not _is_suppress(text):
            self._add_item(lambda: self._format_text(text))

    def add_usage(self, usage: str | None, actions: list[Action],
                  groups: list[_MutuallyExclusiveGroup],
                  prefix: str | None = None) -> None:
        if usage is None or not _is_suppress(usage):
            self._add_item(lambda: self._format_usage(usage, actions, groups,
                                                      prefix))

    def add_argument(self, action: Action) -> None:
        if not _is_suppress(action.help):

            # find all invocations
            invocation_lengths = [len(self._format_action_invocation(action)) +
                                  self._current_indent]
            subactions = action._get_subactions()
            if subactions:
                self._indent()
                for subaction in subactions:
                    invocation_lengths.append(
                        len(self._format_action_invocation(subaction)) +
                        self._current_indent)
                self._dedent()

            # update the maximum item length
            action_length = max(invocation_lengths)
            self._action_max_length = max(self._action_max_length,
                                          action_length)

            # add the item to the list
            self._add_item(lambda: self._format_action(action))

    def add_arguments(self, actions: list[Action]) -> None:
        for action in actions:
            self.add_argument(action)

    # =======================
    # Help-formatting methods
    # =======================

    def format_help(self) -> str:
        help = self._root_section.format_help()
        if help:
            help = self._long_break_matcher.sub('\n\n', help)
            help = help.strip('\n') + '\n'
        return help

    def _join_parts(self, part_strings: list[str]) -> str:
        return ''.join([part
                        for part in part_strings
                        if part and not _is_suppress(part)])

    def _get_lines(self, parts: list[str], indent: str, text_width: int,
                   prefix: str | None = None) -> list[str]:
        # helper for wrapping lines
        lines: list[str] = []
        line: list[str] = []
        indent_length = len(indent)
        if prefix is not None:
            line_len = len(prefix) - 1
        else:
            line_len = indent_length - 1
        for part in parts:
            part_len = len(part)
            if line_len + 1 + part_len > text_width and line:
                lines.append(indent + ' '.join(line))
                line = []
                line_len = indent_length - 1
            line.append(part)
            line_len += part_len + 1
        if line:
            lines.append(indent + ' '.join(line))
        if prefix is not None:
            lines[0] = lines[0][indent_length:]
        return lines

    def _format_usage(self, usage: str | None, actions: list[Action],
                      groups: list[_MutuallyExclusiveGroup],
                      prefix: str | None) -> str:
        if prefix is None:
            prefix = 'usage: '

        # if usage is specified, use that
        if usage is not None:
            text = _format_mapping(usage, {"prog": self._prog})

        # if no optionals or positionals are available, usage is just prog
        elif not actions:
            text = self._prog

        # if optionals and positionals are available, calculate usage
        else:
            prog = self._prog

            parts, pos_start = self._get_actions_usage_parts(actions, groups)
            # build full usage string
            text = ' '.join([part for part in [prog] + parts if part])

            # wrap the usage parts if it's too long
            text_width = self._width - self._current_indent
            if len(prefix) + len(text) > text_width:

                # break usage into wrappable parts
                opt_parts = parts[:pos_start]
                pos_parts = parts[pos_start:]

                # if prog is short, follow it with optionals or positionals
                prog_len = len(prog)
                lines: list[str] = []
                if len(prefix) + prog_len <= 0.75 * text_width:
                    indent = ' ' * (len(prefix) + prog_len + 1)
                    if opt_parts:
                        lines = self._get_lines([prog] + opt_parts, indent,
                                                text_width, prefix)
                        lines.extend(self._get_lines(pos_parts, indent,
                                                     text_width))
                    elif pos_parts:
                        lines = self._get_lines([prog] + pos_parts, indent,
                                                text_width, prefix)
                    else:
                        lines = [prog]

                # if prog is long, put it on its own line
                else:
                    indent = ' ' * len(prefix)
                    all_parts = opt_parts + pos_parts
                    lines = self._get_lines(all_parts, indent, text_width)
                    if len(lines) > 1:
                        lines = []
                        lines.extend(self._get_lines(opt_parts, indent,
                                                     text_width))
                        lines.extend(self._get_lines(pos_parts, indent,
                                                     text_width))
                    lines = [prog] + lines

                # join lines into usage
                text = '\n'.join(lines)

        # prefix with 'usage:'
        return prefix + text + '\n\n'

    def _is_long_option(self, string: str) -> bool:
        return len(string) > 2

    def _get_actions_usage_parts(self, actions: list[Action],
                                 groups: list[_MutuallyExclusiveGroup]
                                 ) -> tuple[list[str], int]:
        """Get usage parts with split index for optionals/positionals.

        Returns (parts, pos_start) where pos_start is the index in parts
        where positionals begin.
        This preserves mutually exclusive group formatting across the
        optionals/positionals boundary (gh-75949).
        """
        actions = [action for action in actions if not _is_suppress(action.help)]
        # group actions by mutually exclusive groups
        # ⛔ The group as an index into `groups`, -1 for none: a dict whose
        # values are `Group | None` is refused when one is popped.
        action_groups: dict[Action, int] = {}
        for action in actions:
            action_groups[action] = -1
        for index, group in enumerate(groups):
            for action in group._group_actions:
                if action in action_groups:
                    action_groups[action] = index
        # positional arguments keep their position
        positionals: list[tuple[bool | None, list[Action]]] = []
        for action in actions:
            if not action.option_strings:
                group_index = action_groups.pop(action)
                if group_index >= 0:
                    group = groups[group_index]
                    group_actions: list[Action] = []
                    for action2 in group._group_actions:
                        if action2.option_strings and action2 in action_groups:
                            if action_groups.pop(action2) >= 0:
                                group_actions.append(action2)
                    group_actions.append(action)
                    positionals.append((group.required, group_actions))
                else:
                    positionals.append((None, [action]))
        # the remaining optional arguments are sorted by the position of
        # the first option in the group
        optionals: list[tuple[bool | None, list[Action]]] = []
        for action in actions:
            if action.option_strings and action in action_groups:
                group_index = action_groups.pop(action)
                if group_index >= 0:
                    group = groups[group_index]
                    group_actions = [action]
                    for action2 in group._group_actions:
                        if action2.option_strings and action2 in action_groups:
                            if action_groups.pop(action2) >= 0:
                                group_actions.append(action2)
                    optionals.append((group.required, group_actions))
                else:
                    optionals.append((None, [action]))

        # collect all actions format strings
        parts: list[str] = []
        pos_start = -1
        for i, (required, group_list) in enumerate(optionals + positionals):
            start = len(parts)
            if i == len(optionals):
                pos_start = start
            in_group = len(group_list) > 1
            for action in group_list:
                # produce all arg strings
                if not action.option_strings:
                    default = self._get_default_metavar_for_positional(action)
                    part = self._format_args(action, default)
                    # if it's in a group, strip the outer []
                    if in_group:
                        if part[0] == '[' and part[-1] == ']':
                            part = part[1:-1]

                # produce the first way to invoke the option in brackets
                else:
                    option_string = action.option_strings[0]

                    # if the Optional doesn't take a value, format is:
                    #    -s or --long
                    if action.nargs == 0:
                        part = action.format_usage()

                    # if the Optional takes a value, format is:
                    #    -s ARGS or --long ARGS
                    else:
                        default = self._get_default_metavar_for_optional(action)
                        args_string = self._format_args(action, default)
                        part = option_string + ' ' + args_string

                    # make it look optional if it's not required or in a group
                    if not (action.required or (required is not None and required) or in_group):
                        part = '[%s]' % part

                # add the action string to the list
                parts.append(part)

            if in_group:
                group_required = required is not None and required
                parts[start] = ('(' if group_required else '[') + parts[start]
                for k in range(start, len(parts) - 1):
                    parts[k] += ' |'
                parts[-1] += ')' if group_required else ']'

        if pos_start < 0:
            pos_start = len(parts)
        return parts, pos_start

    def _format_text(self, text: str) -> str:
        if '%(prog)' in text:
            text = _format_mapping(text, {"prog": self._prog})
        text_width = max(self._width - self._current_indent, 11)
        indent = ' ' * self._current_indent
        return self._fill_text(text, text_width, indent) + '\n\n'

    def _format_action(self, action: Action) -> str:
        # determine the required width and the entry label
        help_position = min(self._action_max_length + 2,
                            self._max_help_position)
        help_width = max(self._width - help_position, 11)
        action_width = help_position - self._current_indent - 2
        action_header = self._format_action_invocation(action)
        indent_first = 0

        # no help; start on same line and add a final newline
        if not action.help:
            action_header = ' ' * self._current_indent + action_header + '\n'

        # short action name; start on the same line and pad two spaces
        elif len(action_header) <= action_width:
            action_header = (' ' * self._current_indent +
                             action_header.ljust(action_width) + '  ')
            indent_first = 0

        # long action name; start on the next line
        else:
            action_header = ' ' * self._current_indent + action_header + '\n'
            indent_first = help_position

        # collect the pieces of the action help
        parts = [action_header]

        # if there was help for the action, add lines of help text
        help = action.help
        if help is not None and help.strip():
            help_text = self._expand_help(action)
            if help_text:
                help_lines = self._split_lines(help_text, help_width)
                parts.append(' ' * indent_first + help_lines[0] + '\n')
                for line in help_lines[1:]:
                    parts.append(' ' * help_position + line + '\n')

        # or add a newline if the description doesn't end with one
        elif not action_header.endswith('\n'):
            parts.append('\n')

        # if there are any sub-actions, add their help as well
        subactions = action._get_subactions()
        if subactions:
            self._indent()
            for subaction in subactions:
                parts.append(self._format_action(subaction))
            self._dedent()

        # return a single string
        return self._join_parts(parts)

    def _format_action_invocation(self, action: Action) -> str:
        if not action.option_strings:
            default = self._get_default_metavar_for_positional(action)
            return ' '.join(self._metavar(action, default, 1))

        else:

            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                return ', '.join(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s, --long ARGS
            else:
                default = self._get_default_metavar_for_optional(action)
                args_string = self._format_args(action, default)
                return ', '.join(action.option_strings) + ' ' + args_string

    def _metavar(self, action: Action, default_metavar: str,
                 tuple_size: int) -> list[str]:
        metavar = action.metavar
        choices = action.choices
        if metavar is not None:
            if isinstance(metavar, tuple):
                return list(metavar)
            result = metavar
        elif choices is not None:
            result = '{%s}' % ','.join([str(choice) for choice in choices])
        else:
            result = default_metavar
        return [result] * tuple_size

    def _format_args(self, action: Action, default_metavar: str) -> str:
        nargs = action.nargs
        if nargs is None:
            result = self._metavar(action, default_metavar, 1)[0]
        elif nargs == OPTIONAL:
            result = '[%s]' % self._metavar(action, default_metavar, 1)[0]
        elif nargs == ZERO_OR_MORE:
            metavar = self._metavar(action, default_metavar, 1)
            if len(metavar) == 2:
                result = '[%s [%s ...]]' % (metavar[0], metavar[1])
            else:
                result = '[%s ...]' % metavar[0]
        elif nargs == ONE_OR_MORE:
            metavar = self._metavar(action, default_metavar, 2)
            if len(metavar) != 2:
                raise TypeError("not enough arguments for format string")
            result = '%s [%s ...]' % (metavar[0], metavar[1])
        elif nargs == REMAINDER:
            result = '...'
        elif nargs == PARSER:
            result = '%s ...' % self._metavar(action, default_metavar, 1)[0]
        elif nargs == SUPPRESS:
            result = ''
        elif isinstance(nargs, int):
            metavar = self._metavar(action, default_metavar, nargs)
            if len(metavar) != nargs:
                raise TypeError("not all arguments converted during string formatting")
            result = ' '.join(metavar)
        else:
            raise ValueError("invalid nargs value")
        return result

    def _expand_help(self, action: Action) -> str:
        help_string = self._get_help_string(action)
        if '%' not in help_string:
            return help_string
        params: dict[str, object] = {}
        for name, value in action._get_kwargs():
            if not _is_suppress(value):
                params[name] = value
        type_func = action.type
        if type_func is not None:
            params['type'] = type_func.__name__
        params['prog'] = self._prog
        choices = action.choices
        if choices is not None:
            params['choices'] = ', '.join([str(choice) for choice in choices])
        return _format_mapping(help_string, params)

    def _split_lines(self, text: str, width: int) -> list[str]:
        text = self._whitespace_matcher.sub(' ', text).strip()
        return textwrap.wrap(text, width)

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        text = self._whitespace_matcher.sub(' ', text).strip()
        return textwrap.fill(text, width,
                             initial_indent=indent,
                             subsequent_indent=indent)

    def _get_help_string(self, action: Action) -> str:
        help = action.help
        return help if help is not None else ''

    def _get_default_metavar_for_optional(self, action: Action) -> str:
        return action.dest.upper()

    def _get_default_metavar_for_positional(self, action: Action) -> str:
        return action.dest


class RawDescriptionHelpFormatter(HelpFormatter):
    """Help message formatter which retains any formatting in descriptions.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _fill_text(self, text: str, width: int, indent: str) -> str:
        return ''.join([indent + line for line in text.splitlines(keepends=True)])


class RawTextHelpFormatter(RawDescriptionHelpFormatter):
    """Help message formatter which retains formatting of all help text.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _split_lines(self, text: str, width: int) -> list[str]:
        return text.splitlines()


class ArgumentDefaultsHelpFormatter(HelpFormatter):
    """Help message formatter which adds default values to argument help.

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _get_help_string(self, action: Action) -> str:
        help = action.help
        if help is None:
            help = ''

        if (
            '%(default)' not in help
            and not _is_suppress(action.default)
            and not action.required
        ):
            if action.option_strings or action.nargs in (OPTIONAL, ZERO_OR_MORE):
                help += ' (default: %(default)s)'
        return help


class MetavarTypeHelpFormatter(HelpFormatter):
    """Help message formatter which uses the argument 'type' as the default
    metavar value (instead of the argument 'dest')

    Only the name of this class is considered a public API. All the methods
    provided by the class are considered an implementation detail.
    """

    def _get_default_metavar_for_optional(self, action: Action) -> str:
        type_func = action.type
        if type_func is None:
            raise AttributeError("'NoneType' object has no attribute '__name__'")
        return type_func.__name__

    def _get_default_metavar_for_positional(self, action: Action) -> str:
        type_func = action.type
        if type_func is None:
            raise AttributeError("'NoneType' object has no attribute '__name__'")
        return type_func.__name__


# =====================
# Options and Arguments
# =====================

def _get_action_name(argument: Action | None) -> str | None:
    if argument is None:
        return None
    elif argument.option_strings:
        return '/'.join(argument.option_strings)
    metavar = argument.metavar
    if metavar is not None and not _is_suppress(metavar):
        if not isinstance(metavar, tuple):
            return metavar
        if argument.nargs == ZERO_OR_MORE and len(metavar) == 2:
            return '%s[, %s]' % (metavar[0], metavar[1])
        elif argument.nargs == ONE_OR_MORE:
            return '%s[, %s]' % (metavar[0], metavar[1])
        else:
            return ', '.join(metavar)
    elif not _is_suppress(argument.dest):
        return argument.dest
    choices = argument.choices
    if choices:
        return '{%s}' % ','.join([str(choice) for choice in choices])
    return None


class ArgumentError(Exception):
    """An error from creating or using an argument (optional or positional).

    The string value of this exception is the message, augmented with
    information about the argument that caused it.
    """

    def __init__(self, argument: Action | None, message: str) -> None:
        name = _get_action_name(argument)
        self.argument_name = name if name is not None else ''
        self.message = message

    def __str__(self) -> str:
        if not self.argument_name:
            return self.message
        return 'argument %s: %s' % (self.argument_name, self.message)


class ArgumentTypeError(Exception):
    """An error from trying to convert a command line string to a type."""
    pass


# ==============
# Action classes
# ==============

class Action(object):
    """Information about how to convert command line strings to Python objects.

    Action objects are used by an ArgumentParser to represent the information
    needed to parse a single argument from one or more strings from the
    command line. The keyword arguments to the Action constructor are also
    all attributes of Action instances.

    Keyword Arguments:

        - option_strings -- A list of command-line option strings which
            should be associated with this action.

        - dest -- The name of the attribute to hold the created object(s)

        - nargs -- The number of command-line arguments that should be
            consumed. By default, one argument will be consumed and a single
            value will be produced.  Other values include:
                - N (an integer) consumes N arguments (and produces a list)
                - '?' consumes zero or one arguments
                - '*' consumes zero or more arguments (and produces a list)
                - '+' consumes one or more arguments (and produces a list)
            Note that the difference between the default and nargs=1 is that
            with the default, a single value will be produced, while with
            nargs=1, a list containing a single value will be produced.

        - const -- The value to be produced if the option is specified and the
            option uses an action that takes no values.

        - default -- The value to be produced if the option is not specified.

        - type -- A callable that accepts a single string argument, and
            returns the converted value.  The standard Python types str, int,
            float, and complex are useful examples of such callables.  If None,
            str is used.

        - choices -- A container of values that should be allowed. If not None,
            after a command-line argument has been converted to the appropriate
            type, an exception will be raised if it is not a member of this
            collection.

        - required -- True if the action must always be specified at the
            command line. This is only meaningful for optional command-line
            arguments.

        - help -- The help string describing the argument.

        - metavar -- The name to be used for the option's argument with the
            help string. If None, the 'dest' value will be used as the name.
    """

    def __init__(self,
                 option_strings: list[str],
                 dest: str,
                 nargs: int | str | None = None,
                 const: object = None,
                 default: object = None,
                 type: Callable[[str], object] | None = None,
                 choices: list[object] | None = None,
                 required: bool = False,
                 help: str | None = None,
                 metavar: str | tuple[str, ...] | None = None,
                 deprecated: bool = False,
                 version: str | None = None) -> None:
        self.option_strings = option_strings
        self.dest = dest
        self.nargs = nargs
        self.const = const
        self.default = default
        self.type = type
        self.choices = choices
        self.required = required
        self.help = help
        self.metavar = metavar
        self.deprecated = deprecated
        self.container: _ActionsContainer | None = None

    def _get_kwargs(self) -> list[tuple[str, object]]:
        return [
            ('option_strings', self.option_strings),
            ('dest', self.dest),
            ('nargs', self.nargs),
            ('const', self.const),
            ('default', self.default),
            ('type', self.type),
            ('choices', self.choices),
            ('required', self.required),
            ('help', self.help),
            ('metavar', self.metavar),
            ('deprecated', self.deprecated),
        ]

    def __repr__(self) -> str:
        arg_strings: list[str] = []
        for name, value in self._get_kwargs():
            if name == 'type':
                type_func = self.type
                arg_strings.append('type=%s' % ('None' if type_func is None
                                                 else type_func.__name__))
            else:
                arg_strings.append('%s=%r' % (name, value))
        return '%s(%s)' % (type(self).__name__, ', '.join(arg_strings))

    def format_usage(self) -> str:
        return self.option_strings[0]

    def _get_subactions(self) -> list[Action]:
        return []

    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: object, option_string: str | None = None) -> None:
        raise NotImplementedError('.__call__() not defined')


class BooleanOptionalAction(Action):
    def __init__(self,
                 option_strings: list[str],
                 dest: str,
                 nargs: int | str | None = None,
                 const: object = None,
                 default: object = None,
                 type: Callable[[str], object] | None = None,
                 choices: list[object] | None = None,
                 required: bool = False,
                 help: str | None = None,
                 metavar: str | tuple[str, ...] | None = None,
                 deprecated: bool = False,
                 version: str | None = None) -> None:

        _option_strings: list[str] = []
        for option_string in option_strings:
            _option_strings.append(option_string)

            if option_string.startswith('--'):
                if option_string.startswith('--no-'):
                    raise ValueError(f'invalid option name {option_string!r} '
                                     f'for BooleanOptionalAction')
                option_string = '--no-' + option_string[2:]
                _option_strings.append(option_string)

        Action.__init__(
            self,
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            required=required,
            help=help,
            deprecated=deprecated)

    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: object, option_string: str | None = None) -> None:
        if option_string is not None and option_string in self.option_strings:
            namespace._set(self.dest, not option_string.startswith('--no-'))

    def format_usage(self) -> str:
        return ' | '.join(self.option_strings)


class _StoreAction(Action):

    def __init__(self,
                 option_strings: list[str],
                 dest: str,
                 nargs: int | str | None = None,
                 const: object = None,
                 default: object = None,
                 type: Callable[[str], object] | None = None,
                 choices: list[object] | None = None,
                 required: bool = False,
                 help: str | None = None,
                 metavar: str | tuple[str, ...] | None = None,
                 deprecated: bool = False,
                 version: str | None = None) -> None:
        if nargs == 0:
            raise ValueError('nargs for store actions must be != 0; if you '
                             'have nothing to store, actions such as store '
                             'true or store const may be more appropriate')
        if const is not None and nargs != OPTIONAL:
            raise ValueError('nargs must be %r to supply const' % OPTIONAL)
        Action.__init__(
            self,
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
            deprecated=deprecated)

    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: object, option_string: str | None = None) -> None:
        namespace._set(self.dest, values)


class _StoreConstAction(Action):

    def __init__(self,
                 option_strings: list[str],
                 dest: str,
                 nargs: int | str | None = None,
                 const: object = None,
                 default: object = None,
                 type: Callable[[str], object] | None = None,
                 choices: list[object] | None = None,
                 required: bool = False,
                 help: str | None = None,
                 metavar: str | tuple[str, ...] | None = None,
                 deprecated: bool = False,
                 version: str | None = None) -> None:
        Action.__init__(
            self,
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=const,
            default=default,
            required=required,
            help=help,
            deprecated=deprecated)

    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: object, option_string: str | None = None) -> None:
        namespace._set(self.dest, self.const)


class _AppendAction(Action):

    def __init__(self,
                 option_strings: list[str],
                 dest: str,
                 nargs: int | str | None = None,
                 const: object = None,
                 default: object = None,
                 type: Callable[[str], object] | None = None,
                 choices: list[object] | None = None,
                 required: bool = False,
                 help: str | None = None,
                 metavar: str | tuple[str, ...] | None = None,
                 deprecated: bool = False,
                 version: str | None = None) -> None:
        if nargs == 0:
            raise ValueError('nargs for append actions must be != 0; if arg '
                             'strings are not supplying the value to append, '
                             'the append const action may be more appropriate')
        if const is not None and nargs != OPTIONAL:
            raise ValueError('nargs must be %r to supply const' % OPTIONAL)
        Action.__init__(
            self,
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
            deprecated=deprecated)

    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: object, option_string: str | None = None) -> None:
        items = _copy_items(namespace._get(self.dest))
        items.append(values)
        namespace._set(self.dest, items)


class _AppendConstAction(Action):

    def __init__(self,
                 option_strings: list[str],
                 dest: str,
                 nargs: int | str | None = None,
                 const: object = None,
                 default: object = None,
                 type: Callable[[str], object] | None = None,
                 choices: list[object] | None = None,
                 required: bool = False,
                 help: str | None = None,
                 metavar: str | tuple[str, ...] | None = None,
                 deprecated: bool = False,
                 version: str | None = None) -> None:
        Action.__init__(
            self,
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=const,
            default=default,
            required=required,
            help=help,
            metavar=metavar,
            deprecated=deprecated)

    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: object, option_string: str | None = None) -> None:
        items = _copy_items(namespace._get(self.dest))
        items.append(self.const)
        namespace._set(self.dest, items)


class _CountAction(Action):

    def __init__(self,
                 option_strings: list[str],
                 dest: str,
                 nargs: int | str | None = None,
                 const: object = None,
                 default: object = None,
                 type: Callable[[str], object] | None = None,
                 choices: list[object] | None = None,
                 required: bool = False,
                 help: str | None = None,
                 metavar: str | tuple[str, ...] | None = None,
                 deprecated: bool = False,
                 version: str | None = None) -> None:
        Action.__init__(
            self,
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            default=default,
            required=required,
            help=help,
            deprecated=deprecated)

    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: object, option_string: str | None = None) -> None:
        count = namespace._get(self.dest)
        if count is None:
            namespace._set(self.dest, 1)
        elif isinstance(count, int):
            namespace._set(self.dest, count + 1)
        else:
            raise TypeError('can only count from an int default')


class _HelpAction(Action):

    def __init__(self,
                 option_strings: list[str],
                 dest: str = SUPPRESS,
                 nargs: int | str | None = None,
                 const: object = None,
                 default: object = SUPPRESS,
                 type: Callable[[str], object] | None = None,
                 choices: list[object] | None = None,
                 required: bool = False,
                 help: str | None = None,
                 metavar: str | tuple[str, ...] | None = None,
                 deprecated: bool = False,
                 version: str | None = None) -> None:
        Action.__init__(
            self,
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help,
            deprecated=deprecated)

    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: object, option_string: str | None = None) -> None:
        parser.print_help()
        parser.exit()


class _VersionAction(Action):

    def __init__(self,
                 option_strings: list[str],
                 dest: str = SUPPRESS,
                 nargs: int | str | None = None,
                 const: object = None,
                 default: object = SUPPRESS,
                 type: Callable[[str], object] | None = None,
                 choices: list[object] | None = None,
                 required: bool = False,
                 help: str | None = None,
                 metavar: str | tuple[str, ...] | None = None,
                 deprecated: bool = False,
                 version: str | None = None) -> None:
        if help is None:
            help = "show program's version number and exit"
        Action.__init__(
            self,
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)
        self.version = version

    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: object, option_string: str | None = None) -> None:
        version = self.version
        if version is None:
            version = parser.version
        formatter = parser._get_formatter()
        formatter.add_text(version)
        parser._print_message(formatter.format_help(), _sys.stdout)
        parser.exit()


class _ChoicesPseudoAction(Action):
    """The help line of one subcommand. Built with the keyword arguments
    every Action takes, as a `type[Action]` construction may build any of
    them; `_make_choice` spells CPython's (name, aliases, help)."""

    def __init__(self,
                 option_strings: list[str],
                 dest: str,
                 nargs: int | str | None = None,
                 const: object = None,
                 default: object = None,
                 type: Callable[[str], object] | None = None,
                 choices: list[object] | None = None,
                 required: bool = False,
                 help: str | None = None,
                 metavar: str | tuple[str, ...] | None = None,
                 deprecated: bool = False,
                 version: str | None = None) -> None:
        Action.__init__(self, option_strings=[], dest=dest, help=help,
                        metavar=metavar)


def _make_choice(name: str, aliases: list[str],
                 help: str | None) -> _ChoicesPseudoAction:
    metavar = dest = name
    if aliases:
        metavar += ' (%s)' % ', '.join(aliases)
    return _ChoicesPseudoAction([], dest, help=help, metavar=metavar)


class _SubParsersAction(Action):
    """`add_subparsers`' action. Built with the keyword arguments every
    Action takes, as a `type[Action]` construction may build any of them;
    the program name and parser class are set after (`_setup`)."""

    def __init__(self,
                 option_strings: list[str],
                 dest: str,
                 nargs: int | str | None = None,
                 const: object = None,
                 default: object = None,
                 type: Callable[[str], object] | None = None,
                 choices: list[object] | None = None,
                 required: bool = False,
                 help: str | None = None,
                 metavar: str | tuple[str, ...] | None = None,
                 deprecated: bool = False,
                 version: str | None = None) -> None:

        self._prog_prefix = ''
        self._parser_class: type[ArgumentParser] = ArgumentParser
        self._name_parser_map: dict[str, ArgumentParser] = {}
        self._names: list[object] = []
        self._choices_actions: list[Action] = []
        self._deprecated: set[str] = set()

        Action.__init__(
            self,
            option_strings=option_strings,
            dest=dest,
            nargs=PARSER,
            choices=self._names,
            required=required,
            help=help,
            metavar=metavar)

    def _setup(self, prog: str, parser_class: type[ArgumentParser]) -> None:
        self._prog_prefix = prog
        self._parser_class = parser_class

    def add_parser(self, name: str, *, deprecated: bool = False,
                   help: str | None = None, aliases: list[str] | None = None,
                   prog: str | None = None, usage: str | None = None,
                   description: str | None = None, epilog: str | None = None,
                   formatter_class: type[HelpFormatter] = HelpFormatter,
                   prefix_chars: str = '-', argument_default: object = None,
                   conflict_handler: str = 'error', add_help: bool = True,
                   allow_abbrev: bool = True,
                   exit_on_error: bool = True) -> ArgumentParser:
        # set prog from the existing prefix
        if prog is None:
            prog = '%s %s' % (self._prog_prefix, name)

        names = aliases if aliases is not None else []
        if name in self._name_parser_map:
            raise ValueError(f'conflicting subparser: {name}')
        for alias in names:
            if alias in self._name_parser_map:
                raise ValueError(f'conflicting subparser alias: {alias}')

        # create a pseudo-action to hold the choice help
        choice_action: Action | None = None
        if help is not None:
            choice_action = _make_choice(name, names, help)
            self._choices_actions.append(choice_action)

        # create the parser and add it to the map
        parser = self._parser_class(
            prog=prog, usage=usage, description=description, epilog=epilog,
            formatter_class=formatter_class, prefix_chars=prefix_chars,
            argument_default=argument_default,
            conflict_handler=conflict_handler, add_help=add_help,
            allow_abbrev=allow_abbrev, exit_on_error=exit_on_error)
        if choice_action is not None:
            parser._check_help(choice_action)
        self._name_parser_map[name] = parser
        self._names.append(name)

        # make parser available under aliases also
        for alias in names:
            self._name_parser_map[alias] = parser
            self._names.append(alias)

        if deprecated:
            self._deprecated.add(name)
            self._deprecated.update(names)

        return parser

    def _get_subactions(self) -> list[Action]:
        return self._choices_actions

    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: object, option_string: str | None = None) -> None:
        if not isinstance(values, list):
            raise TypeError('subparser values must be a list')
        parser_name = str(values[0])
        arg_strings = [str(value) for value in values[1:]]

        # set the parser name if requested
        if not _is_suppress(self.dest):
            namespace._set(self.dest, parser_name)

        # select the parser
        if parser_name not in self._name_parser_map:
            msg = 'unknown parser %r (choices: %s)' % (
                parser_name, ', '.join(self._name_parser_map))
            raise ArgumentError(self, msg)
        subparser = self._name_parser_map[parser_name]

        if parser_name in self._deprecated:
            parser._warning("command '%s' is deprecated" % parser_name)

        # parse all the remaining options into the namespace
        # store any unrecognized options on the object, so that the top
        # level parser can decide what to do with them

        # In case this subparser defines new defaults, we parse them
        # in a new namespace object and then update the original
        # namespace for the relevant parts.
        subnamespace, rest = subparser.parse_known_args(arg_strings, None)
        for key, value in subnamespace._get_kwargs():
            namespace._set(key, value)

        if rest:
            unrecognized = namespace._get(_UNRECOGNIZED_ARGS_ATTR)
            extended: list[object] = []
            # ⛔ Copied, not rebound: the narrowed list rebound to a merged
            # local is refused by the ownership check.
            if isinstance(unrecognized, list):
                for item in unrecognized:
                    extended.append(item)
            for arg in rest:
                extended.append(arg)
            namespace._set(_UNRECOGNIZED_ARGS_ATTR, extended)


class _ExtendAction(_AppendAction):
    def __call__(self, parser: ArgumentParser, namespace: Namespace,
                 values: object, option_string: str | None = None) -> None:
        items = _copy_items(namespace._get(self.dest))
        if isinstance(values, list):
            items.extend(values)
        else:
            raise TypeError("'extend' needs a list of values")
        namespace._set(self.dest, items)


def _copy_items(items: object) -> list[object]:
    if items is None:
        return []
    if isinstance(items, list):
        return items[:]
    raise TypeError('an append action needs a list default')


# ===========================
# Optional and Positional Parsing
# ===========================

class Namespace(object):
    """Simple object for storing attributes.

    Implements equality by attribute names and values, and provides a simple
    string representation.
    """

    def __init__(self, **kwargs: object) -> None:
        self._values: dict[str, object] = {}
        for name in kwargs:
            self._values[name] = kwargs[name]

    def __getattr__(self, name: str) -> object:
        if name in self._values:
            return self._values[name]
        raise AttributeError("'Namespace' object has no attribute %r" % name)

    def _set(self, name: str, value: object) -> None:
        self._values[name] = value

    def _get(self, name: str) -> object:
        return self._values.get(name, None)

    def _has(self, name: str) -> bool:
        return name in self._values

    def _delete(self, name: str) -> None:
        del self._values[name]

    def _get_kwargs(self) -> list[tuple[str, object]]:
        return list(self._values.items())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Namespace):
            return False
        return self._values == other._values

    def __contains__(self, key: str) -> bool:
        return key in self._values

    def __repr__(self) -> str:
        arg_strings: list[str] = []
        star_args: dict[str, object] = {}
        for name, value in self._get_kwargs():
            if name.isidentifier():
                arg_strings.append('%s=%r' % (name, value))
            else:
                star_args[name] = value
        if star_args:
            arg_strings.append('**%s' % repr(star_args))
        return '%s(%s)' % (type(self).__name__, ', '.join(arg_strings))


class _ActionsContainer(object):

    def __init__(self,
                 description: str | None,
                 prefix_chars: str,
                 argument_default: object,
                 conflict_handler: str) -> None:
        self.description = description
        self.argument_default = argument_default
        self.prefix_chars = prefix_chars
        self.conflict_handler = conflict_handler

        # set up registries
        self._registries: dict[str, type[Action]] = {}

        # raise an exception if the conflict handler is invalid
        if conflict_handler not in ('error', 'resolve'):
            raise ValueError(
                f'invalid conflict_resolution value: {conflict_handler!r}')

        # action storage
        self._actions: list[Action] = []
        self._option_string_actions: dict[str, Action] = {}

        # groups
        self._action_groups: list[_ArgumentGroup] = []
        self._mutually_exclusive_groups: list[_MutuallyExclusiveGroup] = []

        # defaults storage
        self._defaults: dict[str, object] = {}

        # determines whether an "option" looks like a negative number
        self._negative_number_matcher = _re.compile(r'-\.?\d')

        # whether or not there are any optionals that look like negative
        # numbers -- uses a list so it can be shared and edited
        self._has_negative_number_optionals: list[bool] = []

    # ====================
    # Registration methods
    # ====================

    def register(self, registry_name: str, value: str,
                 object: type[Action]) -> None:
        if registry_name != 'action':
            raise ValueError('only actions can be registered here')
        self._registries[value] = object

    # ==================================
    # Namespace default accessor methods
    # ==================================

    def set_defaults(self, **kwargs: object) -> None:
        for key in kwargs:
            self._defaults[key] = kwargs[key]

        # if these defaults match any existing arguments, replace
        # the previous default on the object with the new one
        for action in self._actions:
            if action.dest in kwargs:
                action.default = kwargs[action.dest]

    def get_default(self, dest: str) -> object:
        for action in self._actions:
            if action.dest == dest and action.default is not None:
                return action.default
        return self._defaults.get(dest, None)


    # =======================
    # Adding argument actions
    # =======================

    def add_argument(self, *args: str,
                     action: str | type[Action] | None = None,
                     nargs: int | str | None = None,
                     const: object = None,
                     default: object = SUPPRESS + '?unset',
                     type: Callable[[str], object] | None = None,
                     choices: Iterable[object] = (),
                     required: bool | None = None,
                     help: str | None = None,
                     metavar: str | tuple[str, ...] | None = None,
                     dest: str | None = None,
                     version: str | None = None,
                     deprecated: bool = False) -> Action:
        """
        add_argument(dest, ..., name=value, ...)
        add_argument(option_string, option_string, ..., name=value, ...)
        """
        # ⛔ The body is `_add_argument`'s, with the names as a list: a
        # variadic method is copied into every call, and a parser's calls
        # copied this whole body into one function apiece, past what the
        # ownership check can explore.
        choice_list: list[object] | None = list(choices)
        if not choice_list:
            choice_list = None
        return self._add_argument(list(args), action, nargs, const, default,
                                  type, choice_list, required, help, metavar,
                                  dest, version, deprecated)

    def _add_argument(self, args: list[str],
                      action: str | type[Action] | None,
                      nargs: int | str | None,
                      const: object,
                      default: object,
                      type: Callable[[str], object] | None,
                      choice_list: list[object] | None,
                      required: bool | None,
                      help: str | None,
                      metavar: str | tuple[str, ...] | None,
                      dest: str | None,
                      version: str | None,
                      deprecated: bool) -> Action:
        # if no positional args are supplied or only one is supplied and
        # it doesn't look like an option string, parse a positional
        # argument
        chars = self.prefix_chars
        option_strings: list[str] = []
        if not args or len(args) == 1 and args[0][0] not in chars:
            if args and dest is not None:
                raise TypeError('dest supplied twice for positional argument,'
                                ' did you mean metavar?')
            if required is not None:
                msg = "'required' is an invalid argument for positionals"
                raise TypeError(msg)
            if nargs == 0:
                raise ValueError('nargs for positionals must be != 0')
            if not args:
                raise TypeError("add_argument() missing 1 required positional argument: 'dest'")
            dest = args[0]
            # mark positional arguments as required if at least one is
            # always required
            if nargs not in [OPTIONAL, ZERO_OR_MORE, REMAINDER, SUPPRESS]:
                required = True

        # otherwise, we're adding an optional argument
        else:
            dest = self._get_optional_dest(args, dest, option_strings)

        # if no default was supplied, use the parser-level default
        if isinstance(default, str) and default == SUPPRESS + '?unset':
            if dest in self._defaults:
                default = self._defaults[dest]
            elif self.argument_default is not None:
                default = self.argument_default
            elif action in ('store_true',):
                default = False
            elif action in ('store_false',):
                default = True
            elif action in ('help', 'version'):
                default = SUPPRESS
            else:
                default = None

        # create the action object, and add it to the parser
        made = self._make_action(action, option_strings, dest, nargs, const,
                                 default, type, choice_list, required, help,
                                 metavar, version, deprecated)

        # raise an error if action for positional argument does not
        # consume arguments
        if not made.option_strings and made.nargs == 0:
            raise ValueError(f'action {action!r} is not valid for positional arguments')

        # raise an error if the metavar does not match the type
        formatter = self._get_validation_formatter()
        try:
            formatter._format_args(made, '')
        except TypeError:
            raise ValueError("length of metavar tuple does not match nargs")
        self._check_help(made)
        return self._add_action(made)

    def _make_action(self, action: str | type[Action] | None,
                     option_strings: list[str], dest: str,
                     nargs: int | str | None, const: object, default: object,
                     type: Callable[[str], object] | None,
                     choice_list: list[object] | None, required: bool | None,
                     help: str | None,
                     metavar: str | tuple[str, ...] | None,
                     version: str | None, deprecated: bool) -> Action:
        is_required = required is not None and required
        cls: type[Action] = _StoreAction
        if isinstance(action, str):
            if action in self._registries:
                cls = self._registries[action]
            elif action == 'store':
                cls = _StoreAction
            elif action in ('store_const', 'store_true', 'store_false'):
                cls = _StoreConstAction
                if action == 'store_true':
                    const = True
                elif action == 'store_false':
                    const = False
            elif action == 'append':
                cls = _AppendAction
            elif action == 'append_const':
                cls = _AppendConstAction
            elif action == 'count':
                cls = _CountAction
            elif action == 'help':
                cls = _HelpAction
            elif action == 'version':
                cls = _VersionAction
            elif action == 'extend':
                cls = _ExtendAction
            else:
                raise ValueError(f'unknown action {action!r}')
        elif action is not None:
            cls = action
        if cls is _HelpAction:
            return _HelpAction(option_strings, dest, default=default,
                               help=help, deprecated=deprecated)
        if cls is _VersionAction:
            return _VersionAction(option_strings, dest, default=default,
                                  help=help, deprecated=deprecated,
                                  version=version)
        return cls(option_strings=option_strings, dest=dest, nargs=nargs,
                   const=const, default=default, type=type,
                   choices=choice_list, required=is_required, help=help,
                   metavar=metavar, deprecated=deprecated, version=version)

    def add_argument_group(self, title: str | None = None,
                           description: str | None = None) -> _ArgumentGroup:
        group = _ArgumentGroup(self, title, description)
        self._action_groups.append(group)
        return group

    def add_mutually_exclusive_group(self, *, required: bool = False) -> _MutuallyExclusiveGroup:
        group = _MutuallyExclusiveGroup(self, required)
        self._mutually_exclusive_groups.append(group)
        return group

    def _add_action(self, action: Action) -> Action:
        # resolve any conflicts
        self._check_conflict(action)

        # add to actions list
        self._actions.append(action)
        action.container = self

        # index the action by any option strings it has
        for option_string in action.option_strings:
            self._option_string_actions[option_string] = action

        # set the flag if any option strings look like negative numbers
        for option_string in action.option_strings:
            if self._negative_number_matcher.match(option_string) is not None:
                if not self._has_negative_number_optionals:
                    self._has_negative_number_optionals.append(True)

        # return the created action
        return action

    def _remove_action(self, action: Action) -> None:
        self._actions.remove(action)

    def _get_optional_dest(self, args: list[str], dest: str | None,
                           option_strings: list[str]) -> str:
        # determine short and long option strings
        long_option_strings: list[str] = []
        for option_string in args:
            # error on strings that don't start with an appropriate prefix
            if not option_string[0] in self.prefix_chars:
                raise ValueError(
                    f'invalid option string {option_string!r}: '
                    f'must start with a character {self.prefix_chars!r}')

            # strings starting with two prefix characters are long options
            option_strings.append(option_string)
            if len(option_string) > 1 and option_string[1] in self.prefix_chars:
                long_option_strings.append(option_string)

        # infer destination, '--foo-bar' -> 'foo_bar' and '-x' -> 'x'
        if dest is None:
            if long_option_strings:
                dest_option_string = long_option_strings[0]
            else:
                dest_option_string = option_strings[0]
            dest = dest_option_string.lstrip(self.prefix_chars)
            if not dest:
                msg = f'dest= is required for options like {dest_option_string!r}'
                raise TypeError(msg)
            dest = dest.replace('-', '_')

        return dest

    def _check_conflict(self, action: Action) -> None:

        # find all options that conflict with this option
        confl_optionals: list[tuple[str, Action]] = []
        for option_string in action.option_strings:
            if option_string in self._option_string_actions:
                confl_optional = self._option_string_actions[option_string]
                confl_optionals.append((option_string, confl_optional))

        # resolve any conflicts
        if confl_optionals:
            if self.conflict_handler == 'resolve':
                self._handle_conflict_resolve(action, confl_optionals)
            else:
                self._handle_conflict_error(action, confl_optionals)

    def _handle_conflict_error(self, action: Action,
                               conflicting_actions: list[tuple[str, Action]]) -> None:
        conflict_string = ', '.join([option_string
                                     for option_string, conflicting
                                     in conflicting_actions])
        if len(conflicting_actions) == 1:
            message = 'conflicting option string: %s' % conflict_string
        else:
            message = 'conflicting option strings: %s' % conflict_string
        raise ArgumentError(action, message)

    def _handle_conflict_resolve(self, action: Action,
                                 conflicting_actions: list[tuple[str, Action]]) -> None:

        # remove all conflicting options
        for option_string, conflicting in conflicting_actions:

            # remove the conflicting option
            conflicting.option_strings.remove(option_string)
            self._option_string_actions.pop(option_string, None)

            # if the option now has no option string, remove it from the
            # container holding it
            if not conflicting.option_strings:
                container = conflicting.container
                if container is not None:
                    container._remove_action(conflicting)

    def _get_validation_formatter(self) -> HelpFormatter:
        return HelpFormatter('')

    def _check_help(self, action: Action) -> None:
        if action.help:
            formatter = self._get_validation_formatter()
            try:
                formatter._expand_help(action)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError('badly formed help string') from exc


class _ArgumentGroup(_ActionsContainer):

    def __init__(self, container: _ActionsContainer, title: str | None = None,
                 description: str | None = None) -> None:
        _ActionsContainer.__init__(
            self, description=description,
            prefix_chars=container.prefix_chars,
            argument_default=container.argument_default,
            conflict_handler=container.conflict_handler)

        # group attributes
        self.title = title
        self._group_actions: list[Action] = []

        # share most attributes with the container
        self._registries = container._registries
        self._actions = container._actions
        self._option_string_actions = container._option_string_actions
        self._defaults = container._defaults
        self._has_negative_number_optionals = \
            container._has_negative_number_optionals
        self._mutually_exclusive_groups = container._mutually_exclusive_groups

    def _add_action(self, action: Action) -> Action:
        action = _ActionsContainer._add_action(self, action)
        self._group_actions.append(action)
        return action

    def _remove_action(self, action: Action) -> None:
        _ActionsContainer._remove_action(self, action)
        self._group_actions.remove(action)

    def add_argument_group(self, title: str | None = None,
                           description: str | None = None) -> _ArgumentGroup:
        raise ValueError('argument groups cannot be nested')


class _MutuallyExclusiveGroup(_ArgumentGroup):

    def __init__(self, container: _ActionsContainer,
                 required: bool = False) -> None:
        _ArgumentGroup.__init__(self, container)
        self.required = required
        self._container = container

    def _add_action(self, action: Action) -> Action:
        if action.required:
            msg = 'mutually exclusive arguments must be optional'
            raise ValueError(msg)
        action = self._container._add_action(action)
        self._group_actions.append(action)
        return action

    def _remove_action(self, action: Action) -> None:
        self._container._remove_action(action)
        self._group_actions.remove(action)

    def add_mutually_exclusive_group(self, *, required: bool = False) -> _MutuallyExclusiveGroup:
        raise ValueError('mutually exclusive groups cannot be nested')


def _prog_name(prog: str | None = None) -> str:
    if prog is not None:
        return prog
    return _os.path.basename(_sys.argv[0])


# One option an argument string can mean: the action, the option string it
# matched, the separator before an explicit argument, and that argument.
_OptionTuple = tuple[Action | None, str, str | None, str | None]


class _ParseRun:
    """The state `_parse_known_args` keeps while it walks one command line
    (CPython's closures `take_action`, `consume_optional` and
    `consume_positionals`, and the lists they share)."""

    def __init__(self, parser: ArgumentParser, arg_strings: list[str],
                 namespace: Namespace) -> None:
        self.parser = parser
        self.arg_strings = arg_strings
        self.namespace = namespace
        self.action_conflicts: dict[Action, list[Action]] = {}
        self.option_string_indices: dict[int, list[_OptionTuple]] = {}
        self.arg_strings_pattern = ''
        self.seen_actions: set[Action] = set()
        self.seen_non_default_actions: set[Action] = set()
        self.warned: set[str] = set()
        self.extras: list[str] = []
        self.extras_pattern: list[str] = []
        self.positionals: list[Action] = parser._get_positional_actions()

    def take_action(self, action: Action, argument_strings: list[str],
                    option_string: str | None = None) -> None:
        self.seen_actions.add(action)
        argument_values = self.parser._get_values(action, argument_strings)

        # error if this argument is not allowed with other previously
        # seen arguments
        if action.option_strings or argument_strings:
            self.seen_non_default_actions.add(action)
            for conflict_action in self.action_conflicts.get(action, []):
                if conflict_action in self.seen_non_default_actions:
                    msg = 'not allowed with argument %s'
                    action_name = _get_action_name(conflict_action)
                    raise ArgumentError(action, msg % action_name)

        # take the action if we didn't receive a SUPPRESS value
        # (e.g. from a default)
        if not _is_suppress(argument_values):
            action(self.parser, self.namespace, argument_values, option_string)

    def consume_optional(self, start_index: int) -> int:
        parser = self.parser
        arg_strings = self.arg_strings

        # get the optional identified at this index
        option_tuples = self.option_string_indices[start_index]
        # if multiple actions match, the option string was ambiguous
        if len(option_tuples) > 1:
            options = ', '.join([option_string
                for action, option_string, sep, explicit_arg in option_tuples])
            raise ArgumentError(None, 'ambiguous option: %s could match %s'
                                % (arg_strings[start_index], options))

        action, option_string, sep, explicit_arg = option_tuples[0]

        # identify additional optionals in the same arg string
        # (e.g. -xyz is the same as -x -y -z if no args are required)
        action_tuples: list[tuple[Action, list[str], str]] = []
        stop = start_index + 1
        while True:

            # if we found no optional action, skip it
            if action is None:
                self.extras.append(arg_strings[start_index])
                self.extras_pattern.append('O')
                return start_index + 1

            # if there is an explicit argument, try to match the
            # optional's string arguments to only this
            if explicit_arg is not None:
                arg_count = parser._match_argument(action, 'A')

                # if the action is a single-dash option and takes no
                # arguments, try to parse more single-dash options out
                # of the tail of the option string
                chars = parser.prefix_chars
                if (
                    arg_count == 0
                    and option_string[1] not in chars
                    and explicit_arg != ''
                ):
                    if (sep is not None and sep != '') or explicit_arg[0] in chars:
                        msg = 'ignored explicit argument %r'
                        raise ArgumentError(action, msg % explicit_arg)
                    action_tuples.append((action, [], option_string))
                    char = option_string[0]
                    option_string = char + explicit_arg[0]
                    optionals_map = parser._option_string_actions
                    if option_string in optionals_map:
                        action = optionals_map[option_string]
                        explicit_arg = explicit_arg[1:]
                        if not explicit_arg:
                            sep = None
                            explicit_arg = None
                        elif explicit_arg[0] == '=':
                            sep = '='
                            explicit_arg = explicit_arg[1:]
                        else:
                            sep = ''
                    else:
                        self.extras.append(char + explicit_arg)
                        self.extras_pattern.append('O')
                        stop = start_index + 1
                        break
                # if the action expect exactly one argument, we've
                # successfully matched the option; exit the loop
                elif arg_count == 1:
                    stop = start_index + 1
                    action_tuples.append((action, [explicit_arg], option_string))
                    break

                # error if a double-dash option did not use the
                # explicit argument
                else:
                    msg = 'ignored explicit argument %r'
                    raise ArgumentError(action, msg % explicit_arg)

            # if there is no explicit argument, try to match the
            # optional's string arguments with the following strings
            # if successful, exit the loop
            else:
                start = start_index + 1
                selected_patterns = self.arg_strings_pattern[start:]
                arg_count = parser._match_argument(action, selected_patterns)
                stop = start + arg_count
                action_tuples.append((action, arg_strings[start:stop],
                                      option_string))
                break

        # add the Optional to the list and return the index at which
        # the Optional's string args stopped
        assert action_tuples
        for taken, args, taken_string in action_tuples:
            if taken.deprecated and taken_string not in self.warned:
                parser._warning("option '%s' is deprecated" % taken_string)
                self.warned.add(taken_string)
            self.take_action(taken, args, taken_string)
        return stop

    def consume_positionals(self, start_index: int) -> int:
        parser = self.parser
        # match as many Positionals as possible
        selected_pattern = self.arg_strings_pattern[start_index:]
        arg_counts = parser._match_arguments_partial(self.positionals,
                                                     selected_pattern)

        # slice off the appropriate arg strings for each Positional
        # and add the Positional and its args to the list
        for action, arg_count in zip(self.positionals, arg_counts):
            args = self.arg_strings[start_index: start_index + arg_count]
            # Strip out the first '--' if it is not in REMAINDER arg.
            if action.nargs == PARSER:
                if self.arg_strings_pattern[start_index] == '-':
                    assert args[0] == '--'
                    args.remove('--')
            elif action.nargs != REMAINDER:
                if (self.arg_strings_pattern.find('-', start_index,
                                                  start_index + arg_count) >= 0):
                    args.remove('--')
            start_index += arg_count
            if args and action.deprecated and action.dest not in self.warned:
                parser._warning("argument '%s' is deprecated" % action.dest)
                self.warned.add(action.dest)
            self.take_action(action, args)

        # slice off the Positionals that we just parsed and return the
        # index at which the Positionals' string args stopped
        self.positionals[:] = self.positionals[len(arg_counts):]
        return start_index


class ArgumentParser(_ActionsContainer):
    """Object for parsing command line strings into Python objects.

    Keyword Arguments:
        - prog -- The name of the program (default:
            ``os.path.basename(sys.argv[0])``)
        - usage -- A usage message (default: auto-generated from arguments)
        - description -- A description of what the program does
        - epilog -- Text following the argument descriptions
        - formatter_class -- HelpFormatter class for printing help messages
        - prefix_chars -- Characters that prefix optional arguments
        - argument_default -- The default value for all arguments
        - conflict_handler -- String indicating how to handle conflicts
        - add_help -- Add a -h/-help option
        - allow_abbrev -- Allow long options to be abbreviated unambiguously
        - exit_on_error -- Determines whether or not ArgumentParser exits with
            error info when an error occurs
        - suggest_on_error - Enables suggestions for mistyped argument choices
            and subparser names (default: ``False``)
        - color - Allow color output in help messages (default: ``False``)
    """

    def __init__(self,
                 prog: str | None = None,
                 usage: str | None = None,
                 description: str | None = None,
                 epilog: str | None = None,
                 formatter_class: type[HelpFormatter] = HelpFormatter,
                 prefix_chars: str = '-',
                 argument_default: object = None,
                 conflict_handler: str = 'error',
                 add_help: bool = True,
                 allow_abbrev: bool = True,
                 exit_on_error: bool = True,
                 *,
                 suggest_on_error: bool = False,
                 color: bool = True,
                 ) -> None:
        # ⛔ The body is `_setup`'s: a constructor is copied into every
        # construction, and a program that builds a parser per subcommand
        # copied this one into the same function again and again.
        self._setup(prog, usage, description, epilog, formatter_class,
                    prefix_chars, argument_default, conflict_handler,
                    add_help, allow_abbrev, exit_on_error, suggest_on_error,
                    color)

    def _setup(self, prog: str | None, usage: str | None,
               description: str | None, epilog: str | None,
               formatter_class: type[HelpFormatter], prefix_chars: str,
               argument_default: object, conflict_handler: str,
               add_help: bool, allow_abbrev: bool, exit_on_error: bool,
               suggest_on_error: bool, color: bool) -> None:
        _ActionsContainer.__init__(
            self,
            description=description,
            prefix_chars=prefix_chars,
            argument_default=argument_default,
            conflict_handler=conflict_handler)

        self.prog = _prog_name(prog)
        self.usage = usage
        self.epilog = epilog
        self.formatter_class = formatter_class
        self.add_help = add_help
        self.allow_abbrev = allow_abbrev
        self.exit_on_error = exit_on_error
        self.suggest_on_error = suggest_on_error
        self.color = color
        self.version: str | None = None

        self._positionals = self.add_argument_group('positional arguments')
        self._optionals = self.add_argument_group('options')
        self._subparsers: _ArgumentGroup | None = None

        # add help argument if necessary
        # (using explicit default to override global argument_default)
        default_prefix = '-' if '-' in prefix_chars else prefix_chars[0]
        if self.add_help:
            self.add_argument(
                default_prefix+'h', default_prefix*2+'help',
                action='help', default=SUPPRESS,
                help='show this help message and exit')

    # =======================
    # Pretty __repr__ methods
    # =======================

    def __repr__(self) -> str:
        return ('%s(prog=%r, usage=%r, description=%r, formatter_class=%s, '
                'conflict_handler=%r, add_help=%r)' % (
                    type(self).__name__, self.prog, self.usage,
                    self.description,
                    "<class 'argparse.%s'>" % self.formatter_class.__name__,
                    self.conflict_handler, self.add_help))

    # ==================================
    # Optional/Positional adding methods
    # ==================================

    def add_subparsers(self, *, title: str | None = None,
                       description: str | None = None,
                       prog: str | None = None,
                       parser_class: type[ArgumentParser] | None = None,
                       dest: str = SUPPRESS, required: bool = False,
                       help: str | None = None,
                       metavar: str | None = None) -> _SubParsersAction:
        if self._subparsers is not None:
            raise ValueError('cannot have multiple subparser arguments')

        # add the parser class to the arguments if it's not present
        if parser_class is None:
            parser_class = type(self)

        if title is not None or description is not None:
            self._subparsers = self.add_argument_group(
                title if title is not None else 'subcommands', description)
        else:
            self._subparsers = self._positionals

        # prog defaults to the usage message of this parser, skipping
        # optional arguments and with no "usage:" prefix
        if prog is None:
            formatter = self._get_formatter()
            positionals = self._get_positional_actions()
            groups = self._mutually_exclusive_groups
            formatter.add_usage(None, positionals, groups, '')
            prog = formatter.format_help().strip()

        # create the parsers action and add it to the positionals list
        no_options: list[str] = []
        action = _SubParsersAction(option_strings=no_options, dest=dest,
                                   required=required, help=help,
                                   metavar=metavar)
        action._setup(prog, parser_class)
        self._check_help(action)
        subparsers = self._subparsers
        if subparsers is not None:
            subparsers._add_action(action)

        # return the created parsers action
        return action

    def _add_action(self, action: Action) -> Action:
        if action.option_strings:
            self._optionals._add_action(action)
        else:
            self._positionals._add_action(action)
        return action

    def _get_optional_actions(self) -> list[Action]:
        return [action
                for action in self._actions
                if action.option_strings]

    def _get_positional_actions(self) -> list[Action]:
        return [action
                for action in self._actions
                if not action.option_strings]

    # =====================================
    # Command line argument parsing methods
    # =====================================

    def parse_args(self, args: list[str] | None = None,
                   namespace: Namespace | None = None) -> Namespace:
        parsed, argv = self.parse_known_args(args, namespace)
        if argv:
            msg = 'unrecognized arguments: %s' % ' '.join(argv)
            if self.exit_on_error:
                self.error(msg)
            else:
                raise ArgumentError(None, msg)
        return parsed

    def parse_known_args(self, args: list[str] | None = None,
                         namespace: Namespace | None = None
                         ) -> tuple[Namespace, list[str]]:
        if args is None:
            # args default to the system args
            arg_list = _sys.argv[1:]
        else:
            # make sure that args are mutable
            arg_list = list(args)

        # default Namespace built from parser defaults
        if namespace is None:
            namespace = Namespace()

        # add any action defaults that aren't present
        for action in self._actions:
            if not _is_suppress(action.dest):
                if not namespace._has(action.dest):
                    if not _is_suppress(action.default):
                        namespace._set(action.dest, action.default)

        # add any parser defaults that aren't present
        for dest in self._defaults:
            if not namespace._has(dest):
                namespace._set(dest, self._defaults[dest])

        # parse the arguments and exit if there are any errors
        extras: list[str] = []
        if self.exit_on_error:
            try:
                extras = self._parse_known_args(arg_list, namespace)
            except ArgumentError as err:
                self.error(str(err))
        else:
            extras = self._parse_known_args(arg_list, namespace)

        unrecognized = namespace._get(_UNRECOGNIZED_ARGS_ATTR)
        if isinstance(unrecognized, list):
            for arg in unrecognized:
                extras.append(str(arg))
            namespace._delete(_UNRECOGNIZED_ARGS_ATTR)
        return namespace, extras

    def _parse_known_args(self, arg_strings: list[str],
                          namespace: Namespace) -> list[str]:
        run = _ParseRun(self, arg_strings, namespace)

        # map all mutually exclusive arguments to the other arguments
        # they can't occur with
        for mutex_group in self._mutually_exclusive_groups:
            group_actions = mutex_group._group_actions
            for i, mutex_action in enumerate(mutex_group._group_actions):
                if mutex_action not in run.action_conflicts:
                    run.action_conflicts[mutex_action] = []
                conflicts = run.action_conflicts[mutex_action]
                conflicts.extend(group_actions[:i])
                conflicts.extend(group_actions[i + 1:])

        # find all option indices, and determine the arg_string_pattern
        # which has an 'O' if there is an option at an index,
        # an 'A' if there is an argument, or a '-' if there is a '--'
        arg_string_pattern_parts: list[str] = []
        after_dashes = False
        for i, arg_string in enumerate(arg_strings):

            # all args after -- are non-options
            if after_dashes:
                arg_string_pattern_parts.append('A')
            elif arg_string == '--':
                arg_string_pattern_parts.append('-')
                after_dashes = True

            # otherwise, add the arg to the arg strings
            # and note the index if it was an option
            else:
                option_tuples = self._parse_optional(arg_string)
                if option_tuples is None:
                    pattern = 'A'
                else:
                    run.option_string_indices[i] = option_tuples
                    pattern = 'O'
                arg_string_pattern_parts.append(pattern)

        # join the pieces together to form the pattern
        run.arg_strings_pattern = ''.join(arg_string_pattern_parts)

        # consume Positionals and Optionals alternately, until we have
        # passed the last option string
        start_index = 0
        if run.option_string_indices:
            max_option_string_index = max(run.option_string_indices)
        else:
            max_option_string_index = -1
        while start_index <= max_option_string_index:

            # consume any Positionals preceding the next option
            next_option_string_index = start_index
            while next_option_string_index <= max_option_string_index:
                if next_option_string_index in run.option_string_indices:
                    break
                next_option_string_index += 1
            if start_index != next_option_string_index:
                positionals_end_index = run.consume_positionals(start_index)

                # only try to parse the next optional if we didn't consume
                # the option string during the positionals parsing
                if positionals_end_index > start_index:
                    start_index = positionals_end_index
                    continue
                else:
                    start_index = positionals_end_index

            # if we consumed all the positionals we could and we're not
            # at the index of an option string, there were extra arguments
            if start_index not in run.option_string_indices:
                strings = arg_strings[start_index:next_option_string_index]
                run.extras.extend(strings)
                run.extras_pattern.extend(
                    run.arg_strings_pattern[start_index:next_option_string_index])
                start_index = next_option_string_index

            # consume the next optional and any arguments for it
            start_index = run.consume_optional(start_index)

        # consume any positionals following the last Optional
        stop_index = run.consume_positionals(start_index)

        # if we didn't consume all the argument strings, there were extras
        run.extras.extend(arg_strings[stop_index:])

        # make sure all required actions were present and also convert
        # action defaults which were not given as arguments
        required_actions: list[str] = []
        for action in self._actions:
            if action not in run.seen_actions:
                if action.required:
                    name = _get_action_name(action)
                    required_actions.append(name if name is not None else 'None')
                else:
                    # Convert action default now instead of doing it before
                    # parsing arguments to avoid calling convert functions
                    # twice (which may fail) if the argument was given, but
                    # only if it was defined already in the namespace
                    default = action.default
                    if (isinstance(default, str) and
                        namespace._has(action.dest) and
                        namespace._get(action.dest) == default and
                        not _is_suppress(default)):
                        namespace._set(action.dest,
                                       self._get_value(action, default))

        if required_actions:
            raise ArgumentError(None, 'the following arguments are required: %s' %
                       ', '.join(required_actions))

        # make sure all required groups had one option present
        for group in self._mutually_exclusive_groups:
            if group.required:
                found = False
                for action in group._group_actions:
                    if action in run.seen_non_default_actions:
                        found = True
                        break

                # if no actions were used, report the error
                if not found:
                    names: list[str] = []
                    for action in group._group_actions:
                        if not _is_suppress(action.help):
                            name = _get_action_name(action)
                            names.append(name if name is not None else 'None')
                    msg = 'one of the arguments %s is required'
                    raise ArgumentError(None, msg % ' '.join(names))

        # return the extra arguments
        return run.extras

    def _match_argument(self, action: Action, arg_strings_pattern: str) -> int:
        # match the pattern for this action to the arg strings
        nargs_pattern = self._get_nargs_pattern(action)
        match = _re.match(nargs_pattern, arg_strings_pattern)

        # raise an exception if we weren't able to find a match
        if match is None:
            nargs = action.nargs
            if nargs is None:
                msg = 'expected one argument'
            elif nargs == OPTIONAL:
                msg = 'expected at most one argument'
            elif nargs == ONE_OR_MORE:
                msg = 'expected at least one argument'
            elif isinstance(nargs, int) and nargs == 1:
                msg = 'expected %s argument' % nargs
            else:
                msg = 'expected %s arguments' % nargs
            raise ArgumentError(action, msg)

        # return the number of arguments matched
        group = match.group(1)
        return len(group) if group is not None else 0

    def _match_arguments_partial(self, actions: list[Action],
                                 arg_strings_pattern: str) -> list[int]:
        # progressively shorten the actions list by slicing off the
        # final actions until we find a match
        for i in range(len(actions), 0, -1):
            actions_slice = actions[:i]
            pattern = ''.join([self._get_nargs_pattern(action)
                               for action in actions_slice])
            match = _re.match(pattern, arg_strings_pattern)
            if match is not None:
                result: list[int] = []
                for index in range(1, i + 1):
                    string = match.group(index)
                    result.append(len(string) if string is not None else 0)
                if (match.end() < len(arg_strings_pattern)
                    and arg_strings_pattern[match.end()] == 'O'):
                    while result and result[-1] == 0:
                        del result[-1]
                return result

        # return the list of arg string counts
        return []

    def _parse_optional(self, arg_string: str) -> list[_OptionTuple] | None:
        # if it's an empty string, it was meant to be a positional
        if not arg_string:
            return None

        # if it doesn't start with a prefix, it was meant to be positional
        if not arg_string[0] in self.prefix_chars:
            return None

        # if the option string is present in the parser, return the action
        if arg_string in self._option_string_actions:
            action = self._option_string_actions[arg_string]
            return [(action, arg_string, None, None)]

        # if it's just a single character, it was meant to be positional
        if len(arg_string) == 1:
            return None

        # if the option string before the "=" is present, return the action
        option_string, sep, explicit_arg = arg_string.partition('=')
        if sep and option_string in self._option_string_actions:
            action = self._option_string_actions[option_string]
            return [(action, option_string, sep, explicit_arg)]

        # search through all possible prefixes of the option string
        # and all actions in the parser for possible interpretations
        option_tuples = self._get_option_tuples(arg_string)

        if option_tuples:
            return option_tuples

        # if it was not found as an option, but it looks like a negative
        # number, it was meant to be positional
        # unless there are negative-number-like options
        if self._negative_number_matcher.match(arg_string) is not None:
            if not self._has_negative_number_optionals:
                return None

        # if it contains a space, it was meant to be a positional
        if ' ' in arg_string:
            return None

        # it was meant to be an optional but there is no such option
        # in this parser (though it might be a valid option in a subparser)
        return [(None, arg_string, None, None)]

    def _get_option_tuples(self, option_string: str) -> list[_OptionTuple]:
        result: list[_OptionTuple] = []

        # option strings starting with two prefix characters are only
        # split at the '='
        chars = self.prefix_chars
        if option_string[0] in chars and option_string[1] in chars:
            if self.allow_abbrev:
                option_prefix, sep, explicit_arg = option_string.partition('=')
                found_sep: str | None = sep if sep else None
                found_arg: str | None = explicit_arg if sep else None
                for candidate in self._option_string_actions:
                    if candidate.startswith(option_prefix):
                        action = self._option_string_actions[candidate]
                        result.append((action, candidate, found_sep, found_arg))

        # single character options can be concatenated with their arguments
        # but multiple character options always have to have their argument
        # separate
        elif option_string[0] in chars and option_string[1] not in chars:
            option_prefix, sep, explicit_arg = option_string.partition('=')
            found_sep = sep if sep else None
            found_arg = explicit_arg if sep else None
            short_option_prefix = option_string[:2]
            short_explicit_arg = option_string[2:]

            for candidate in self._option_string_actions:
                if candidate == short_option_prefix:
                    action = self._option_string_actions[candidate]
                    result.append((action, candidate, '', short_explicit_arg))
                elif self.allow_abbrev and candidate.startswith(option_prefix):
                    action = self._option_string_actions[candidate]
                    result.append((action, candidate, found_sep, found_arg))

        # shouldn't ever get here
        else:
            raise ArgumentError(None, 'unexpected option string: %s' % option_string)

        # return the collected option tuples
        return result

    def _get_nargs_pattern(self, action: Action) -> str:
        # in all examples below, we have to allow for '--' args
        # which are represented as '-' in the pattern
        nargs = action.nargs

        # if this is an optional action, -- is not allowed
        option = bool(action.option_strings)

        # the default (None) is assumed to be a single argument
        if nargs is None:
            nargs_pattern = '([A])' if option else '(-*A-*)'

        # allow zero or one arguments
        elif nargs == OPTIONAL:
            nargs_pattern = '(A?)' if option else '(-*A?-*)'

        # allow zero or more arguments
        elif nargs == ZERO_OR_MORE:
            nargs_pattern = '(A*)' if option else '(-*[A-]*)'

        # allow one or more arguments
        elif nargs == ONE_OR_MORE:
            nargs_pattern = '(A+)' if option else '(-*A[A-]*)'

        # allow any number of options or arguments
        elif nargs == REMAINDER:
            nargs_pattern = '([AO]*)' if option else '(.*)'

        # allow one argument followed by any number of options or arguments
        elif nargs == PARSER:
            nargs_pattern = '(A[AO]*)' if option else '(-*A[-AO]*)'

        # suppress action, like nargs=0
        elif nargs == SUPPRESS:
            nargs_pattern = '()' if option else '(-*)'

        # all others should be integers
        elif isinstance(nargs, int):
            nargs_pattern = ('([AO]{%d})' % nargs if option
                             else '((?:-*A){%d}-*)' % nargs)
        else:
            raise ValueError('invalid nargs value')

        # return the pattern
        return nargs_pattern

    # ========================
    # Value conversion methods
    # ========================

    def _get_values(self, action: Action, arg_strings: list[str]) -> object:
        # optional argument produces a default when not present
        value: object = None
        if not arg_strings and action.nargs == OPTIONAL:
            if action.option_strings:
                value = action.const
            else:
                value = action.default
            if isinstance(value, str) and not _is_suppress(value):
                value = self._get_value(action, value)

        # when nargs='*' on a positional, if there were no command-line
        # args, use the default if it is anything other than None
        elif (not arg_strings and action.nargs == ZERO_OR_MORE and
              not action.option_strings):
            if action.default is not None:
                value = action.default
            else:
                empty: list[object] = []
                value = empty

        # single argument or optional argument produces a single value
        elif len(arg_strings) == 1 and (action.nargs is None or action.nargs == OPTIONAL):
            arg_string = arg_strings[0]
            value = self._get_value(action, arg_string)
            self._check_value(action, value)

        # REMAINDER arguments convert all values, checking none
        elif action.nargs == REMAINDER:
            value = [self._get_value(action, v) for v in arg_strings]

        # PARSER arguments convert all values, but check only the first
        elif action.nargs == PARSER:
            values = [self._get_value(action, v) for v in arg_strings]
            self._check_value(action, values[0])
            value = values

        # SUPPRESS argument does not put anything in the namespace
        elif action.nargs == SUPPRESS:
            value = SUPPRESS

        # all other types of nargs produce a list
        else:
            values = [self._get_value(action, v) for v in arg_strings]
            for v in values:
                self._check_value(action, v)
            value = values

        # return the converted value
        return value

    def _get_value(self, action: Action, arg_string: str) -> object:
        type_func = action.type
        if type_func is None:
            return arg_string

        # convert the value to the appropriate type
        try:
            result = type_func(arg_string)

        # ArgumentTypeErrors indicate errors
        except ArgumentTypeError as err:
            msg = str(err)
            raise ArgumentError(action, msg)

        # TypeErrors or ValueErrors also indicate errors
        except (TypeError, ValueError):
            name = type_func.__name__
            msg = 'invalid %s value: %r' % (name, arg_string)
            raise ArgumentError(action, msg)

        # return the converted value
        return result

    def _check_value(self, action: Action, value: object) -> None:
        # converted value must be one of the choices (if specified)
        choices = action.choices
        if choices is None:
            return

        if value not in choices:
            choice_text = ', '.join([repr(str(choice)) for choice in choices])
            msg = 'invalid choice: %r (choose from %s)' % (str(value), choice_text)
            if self.suggest_on_error and isinstance(value, str):
                names: list[str] = []
                for choice in choices:
                    if isinstance(choice, str):
                        names.append(choice)
                if len(names) == len(choices):
                    suggestions = difflib.get_close_matches(value, names, 1)
                    if suggestions:
                        msg = ('invalid choice: %r, maybe you meant %r? '
                               '(choose from %s)' % (value, suggestions[0],
                                                     choice_text))
            raise ArgumentError(action, msg)

    # =======================
    # Help-formatting methods
    # =======================

    def format_usage(self) -> str:
        formatter = self._get_formatter()
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)
        return formatter.format_help()

    def format_help(self) -> str:
        formatter = self._get_formatter()

        # usage
        formatter.add_usage(self.usage, self._actions,
                            self._mutually_exclusive_groups)

        # description
        formatter.add_text(self.description)

        # positionals, optionals and user-defined groups
        for action_group in self._action_groups:
            formatter.start_section(action_group.title)
            formatter.add_text(action_group.description)
            formatter.add_arguments(action_group._group_actions)
            formatter.end_section()

        # epilog
        formatter.add_text(self.epilog)

        # determine help from format above
        return formatter.format_help()

    def _get_formatter(self) -> HelpFormatter:
        return self.formatter_class(prog=self.prog)

    def _get_validation_formatter(self) -> HelpFormatter:
        return self._get_formatter()

    # =====================
    # Help-printing methods
    # =====================

    def print_usage(self, file: io.TextIOWrapper | io.StringIO | None = None) -> None:
        if file is None:
            file = _sys.stdout
        self._print_message(self.format_usage(), file)

    def print_help(self, file: io.TextIOWrapper | io.StringIO | None = None) -> None:
        if file is None:
            file = _sys.stdout
        self._print_message(self.format_help(), file)

    def _print_message(self, message: str,
                       file: io.TextIOWrapper | io.StringIO | None = None) -> None:
        if message:
            if file is None:
                file = _sys.stderr
            try:
                file.write(message)
            except OSError:
                pass

    # ===============
    # Exiting methods
    # ===============

    def exit(self, status: int = 0, message: str | None = None) -> None:
        if message:
            self._print_message(message, _sys.stderr)
        _sys.exit(status)

    def error(self, message: str) -> None:
        """error(message: string)

        Prints a usage message incorporating the message to stderr and
        exits.

        If you override this in a subclass, it should not return -- it
        should either exit or raise an exception.
        """
        self.print_usage(_sys.stderr)
        self.exit(2, '%s: error: %s\n' % (self.prog, message))

    def _warning(self, message: str) -> None:
        self._print_message('%s: warning: %s\n' % (self.prog, message),
                            _sys.stderr)
