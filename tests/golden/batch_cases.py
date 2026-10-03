"""ctest runner for a BATCH of golden cases compiled as one program.

A golden case costs about 0.7 s, and almost none of it is the program: the
compile and the JIT build are a fixed price per process (rfc/test-suite-debt.md).
This runner pays it once for many cases. It joins them into ONE module, each
case still module-level code -- its defs, classes and globals stay where the
case put them -- with every name a case binds at module level renamed
`_c<i>_<name>` so the cases cannot see each other. A marker line printed between
cases cuts the output back into one slice per case, and each slice is compared
with that case's own .stdout.

⛔ Not wrapped in functions: a case's module-level names would become locals and
its defs closures, so the batch would exercise a different path than the one
the case was written for. Renamed module-level code is the same path.

Which cases can join (`eligible`): they expect exit 0 and an exact stdout, have
no .stderr-re and no declared layer, and nothing about them can tell that it was
renamed or that it shares a process -- see `refusal`. A case that cannot join
is not this runner's: ctest registers it on its own (tests/CMakeLists.txt asks
`--list-eligible` at configure time).

What happens when the batch does not simply pass:
  - it does not compile: split in halves and retry, down to single cases, which
    run alone through run_case.py -- so one case the batch cannot hold costs
    only itself;
  - it stops part way (an uncaught exception, a crash): the cases whose closing
    marker was printed are judged, the rest run alone;
  - a case's slice differs from its .stdout: the case runs alone. If it passes
    alone the test still FAILS, as a BATCH-DISAGREE: the same code gave another
    answer because other code ran in the same process or module. That is how
    `round(-15, -1)` was found negating the shared small int 20 for every later
    reader, and dropping it here would be dropping that finding. A case that is
    context-dependent on purpose goes in batch-exclusions.txt with its reason.

Exit 0 when every case passed, alone or in the batch, with no disagreement.
"""

import argparse
import ast
import builtins
import io
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import tokenize
import typing

HERE = pathlib.Path(__file__).resolve().parent
RUN_CASE = HERE / "run_case.py"
EXCLUSIONS = HERE / "batch-exclusions.txt"
LAYERS = HERE / "layers.txt"
MARKER = "\x1elython-batch-case-end"

# Spellings by which a program can observe its module, its process, or the
# text of its own source -- each would read differently from inside a batch.
CONTEXT = re.compile(
    r"\b(exit|quit|input|stdin|__name__|__file__|__doc__|__module__|"
    r"__qualname__|globals|locals|vars|dir|chdir|environ|getenv|putenv|"
    r"unsetenv|getcwd|argv|"
    r"inspect|traceback|setrecursionlimit|_exit|signal|atexit)\b")


# --- which names a case binds at module level ---------------------------------

def bound_names(target: ast.AST) -> "list[str]":
    return [node.id for node in ast.walk(target)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)]


def module_imports(tree: ast.Module) -> "dict[str, str]":
    """Each name a module-level import binds, and what it binds it to."""
    imports: "dict[str, str]" = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                imports[alias.asname or alias.name] = alias.name
        elif isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                imports[alias.asname or alias.name] = (
                    f"{'.' * stmt.level}{stmt.module or ''}.{alias.name}")
    return imports


def module_bindings(tree: ast.Module) -> "set[str] | str":
    """The names the module binds other than by import, or why they cannot
    be known.

    ⛔ Imports are not renamed: Lython recognizes `dataclass`, `Enum`,
    `NamedTuple` and a module's own functions by the spelling the program
    imports them under, so `from dataclasses import dataclass as _c0_dataclass`
    is refused where the case is not. Two cases importing one name share it
    instead, and `compatible` keeps cases that bind it differently apart.
    """
    names: "set[str]" = set()

    def visit(stmts: "list[ast.stmt]") -> "str | None":
        for stmt in stmts:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                names.add(stmt.name)
                continue
            if isinstance(stmt, ast.TypeAlias):
                names.add(stmt.name.id)
                continue
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                for alias in stmt.names:
                    if alias.name == "*":
                        return "a star import"
                    if alias.asname is None and "." in alias.name:
                        return "a dotted import with no `as`"
                continue
            # Module-level control flow binds into the module too; nested
            # defs and classes inside it are found by the walk below.
            for node in ast.walk(stmt):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.add(node.id)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef, ast.Lambda)):
                    if not isinstance(node, ast.Lambda):
                        names.add(node.name)
                    # ⛔ Not descended: a body binds into its own scope.
                elif isinstance(node, ast.NamedExpr):
                    names.add(node.target.id)
                elif isinstance(node, ast.ExceptHandler) and node.name:
                    names.add(node.name)
                elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                    names.add(node.name)
                elif isinstance(node, ast.MatchMapping) and node.rest:
                    names.add(node.rest)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    return "an import inside a statement"
        return None

    why = visit(tree.body)
    if why:
        return why
    rebound = names & set(module_imports(tree))
    if rebound:
        return f"it rebinds the imported name `{sorted(rebound)[0]}`"
    return names


# --- renaming, scope by scope -------------------------------------------------

class Scope:
    def __init__(self, kind: str, local: "set[str]",
                 parent: "Scope | None") -> None:
        self.kind = kind          # module, function, class, comprehension
        self.local = local
        self.parent = parent


def function_locals(node: ast.AST) -> "set[str]":
    """Names a function (or lambda) body binds, minus its global declarations."""
    local: "set[str]" = set()
    declared_global: "set[str]" = set()
    args = node.args  # type: ignore[attr-defined]
    for arg in (args.posonlyargs + args.args + args.kwonlyargs):
        local.add(arg.arg)
    if args.vararg:
        local.add(args.vararg.arg)
    if args.kwarg:
        local.add(args.kwarg.arg)
    body = node.body if isinstance(node.body, list) else [node.body]  # type: ignore[attr-defined]
    stack: "list[ast.AST]" = list(body)
    while stack:
        current = stack.pop()
        if isinstance(current, ast.Global):
            declared_global.update(current.names)
            continue
        if isinstance(current, ast.Nonlocal):
            continue
        if isinstance(current, ast.Name) and isinstance(current.ctx, ast.Store):
            local.add(current.id)
        elif isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef)):
            local.add(current.name)
            # Decorators, defaults and bases evaluate here; the body does not.
            stack.extend(current.decorator_list)
            if isinstance(current, ast.ClassDef):
                stack.extend(current.bases)
                stack.extend(k.value for k in current.keywords)
            else:
                stack.extend(current.args.defaults)
                stack.extend(d for d in current.args.kw_defaults if d)
            continue
        elif isinstance(current, ast.Lambda):
            stack.extend(current.args.defaults)
            continue
        elif isinstance(current, (ast.ListComp, ast.SetComp, ast.DictComp,
                                  ast.GeneratorExp)):
            # Its targets are its own; a walrus inside binds here.
            for sub in ast.walk(current):
                if isinstance(sub, ast.NamedExpr):
                    local.add(sub.target.id)
            stack.append(current.generators[0].iter)
            continue
        elif isinstance(current, (ast.Import, ast.ImportFrom)):
            for alias in current.names:
                local.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(current, ast.ExceptHandler) and current.name:
            local.add(current.name)
        elif isinstance(current, (ast.MatchAs, ast.MatchStar)) and current.name:
            local.add(current.name)
        elif isinstance(current, ast.MatchMapping) and current.rest:
            local.add(current.rest)
        stack.extend(ast.iter_child_nodes(current))
    return local - declared_global


class Edit(typing.NamedTuple):
    """One rename in the source: `old` becomes `new` at a token found in the
    span of `node` -- the token itself for a Name, the one after `after` for
    a name the AST gives no position of (`def f`, `except E as e`, ...), or
    the whole span for a string annotation."""
    node: ast.AST
    old: str
    new: str
    after: "str | None"
    whole: bool = False


class Renamer(ast.NodeTransformer):
    """Finds every reference that resolves to one of the module's names.

    ⛔ The edits are applied to the case's SOURCE, not to its tree: printing a
    tree back (`ast.unparse`) respells it -- `0x_ff` becomes `255`, quotes and
    parentheses are normalized -- and a case about how the parser reads a
    spelling would stop testing it inside a batch, passing all the same.
    """

    def __init__(self, names: "set[str]", prefix: str) -> None:
        self.names = names
        self.prefix = prefix
        self.scope = Scope("module", set(), None)
        self.edits: "list[Edit]" = []
        # Inside a string annotation, whose tree is thrown away: rename the
        # tree itself so it can be printed back into the string.
        self.in_string = False

    def rename(self, node: ast.AST, old: str, after: "str | None") -> str:
        new = self.prefix + old
        if not self.in_string:
            self.edits.append(Edit(node, old, new, after))
        return new

    def resolves_to_module(self, name: str) -> bool:
        scope: "Scope | None" = self.scope
        first = True
        while scope is not None:
            if scope.kind == "module":
                return name in self.names
            # A class body's names are visible in the body itself, never in
            # the functions nested inside it.
            if (scope.kind != "class" or first) and name in scope.local:
                return False
            first = False
            scope = scope.parent
        return False

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self.resolves_to_module(node.id):
            node.id = self.rename(node, node.id, None)
        return node

    def visit_Global(self, node: ast.Global) -> ast.AST:
        node.names = [self.rename(node, n, None) if n in self.names else n
                      for n in node.names]
        return node

    # A string annotation names a class the same way an expression would.
    def annotation(self, node: "ast.expr | None") -> "ast.expr | None":
        if node is None:
            return None
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                try:
                    inner = ast.parse(sub.value, mode="eval")
                except SyntaxError:
                    continue
                saved, self.in_string = self.in_string, True
                text = ast.unparse(self.visit(inner.body))
                self.in_string = saved
                if text != ast.unparse(ast.parse(sub.value, mode="eval").body):
                    self.edits.append(Edit(sub, sub.value, repr(text), None,
                                           whole=True))
                    sub.value = text
        return self.visit(node)

    def enter(self, kind: str, local: "set[str]") -> None:
        self.scope = Scope(kind, local, self.scope)

    def leave(self) -> None:
        assert self.scope.parent is not None
        self.scope = self.scope.parent

    def visit_arguments_in_outer_scope(self, args: ast.arguments) -> None:
        args.defaults = [self.visit(d) for d in args.defaults]
        args.kw_defaults = [self.visit(d) if d else None
                            for d in args.kw_defaults]
        for arg in (args.posonlyargs + args.args + args.kwonlyargs +
                    [a for a in (args.vararg, args.kwarg) if a]):
            arg.annotation = self.annotation(arg.annotation)

    def visit_FunctionDef(self, node: "ast.FunctionDef | ast.AsyncFunctionDef") -> ast.AST:
        if self.resolves_to_module(node.name):
            node.name = self.rename(node, node.name, "def")
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        self.visit_arguments_in_outer_scope(node.args)
        node.returns = self.annotation(node.returns)
        self.enter("function", function_locals(node))
        node.body = [self.visit(s) for s in node.body]
        self.leave()
        return node

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        self.visit_arguments_in_outer_scope(node.args)
        self.enter("function", function_locals(node))
        node.body = self.visit(node.body)
        self.leave()
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        if self.resolves_to_module(node.name):
            node.name = self.rename(node, node.name, "class")
        node.decorator_list = [self.visit(d) for d in node.decorator_list]
        node.bases = [self.visit(b) for b in node.bases]
        node.keywords = [self.visit(k) for k in node.keywords]
        local: "set[str]" = set()
        for stmt in node.body:
            for sub in ast.walk(stmt):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                    local.add(sub.id)
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                local.add(stmt.name)
        self.enter("class", local)
        body = []
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign):
                stmt.annotation = self.annotation(stmt.annotation)  # type: ignore[assignment]
                if stmt.value is not None:
                    stmt.value = self.visit(stmt.value)
                stmt.target = self.visit(stmt.target)
                body.append(stmt)
            else:
                body.append(self.visit(stmt))
        node.body = body
        self.leave()
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        node.annotation = self.annotation(node.annotation)  # type: ignore[assignment]
        if node.value is not None:
            node.value = self.visit(node.value)
        node.target = self.visit(node.target)
        return node

    def comprehension(self, node: ast.AST, parts: "list[str]") -> ast.AST:
        generators = node.generators  # type: ignore[attr-defined]
        # The first iterable is evaluated in the enclosing scope.
        generators[0].iter = self.visit(generators[0].iter)
        local = {n for g in generators for n in bound_names(g.target)}
        self.enter("comprehension", local)
        for index, generator in enumerate(generators):
            generator.target = self.visit(generator.target)
            if index:
                generator.iter = self.visit(generator.iter)
            generator.ifs = [self.visit(i) for i in generator.ifs]
        for part in parts:
            setattr(node, part, self.visit(getattr(node, part)))
        self.leave()
        return node

    def visit_ListComp(self, node: ast.ListComp) -> ast.AST:
        return self.comprehension(node, ["elt"])

    visit_SetComp = visit_ListComp  # type: ignore[assignment]
    visit_GeneratorExp = visit_ListComp  # type: ignore[assignment]

    def visit_DictComp(self, node: ast.DictComp) -> ast.AST:
        return self.comprehension(node, ["key", "value"])

    def visit_NamedExpr(self, node: ast.NamedExpr) -> ast.AST:
        # The target binds in the nearest function or module scope, never in
        # the comprehension it is written in.
        scope = self.scope
        while scope.kind == "comprehension" and scope.parent is not None:
            scope = scope.parent
        saved, self.scope = self.scope, scope
        node.target = self.visit(node.target)
        self.scope = saved
        node.value = self.visit(node.value)
        return node

    def visit_TypeAlias(self, node: ast.TypeAlias) -> ast.AST:
        node.name = self.visit(node.name)  # type: ignore[assignment]
        node.value = self.annotation(node.value)  # type: ignore[assignment]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        if node.type is not None:
            node.type = self.visit(node.type)
        if node.name and self.resolves_to_module(node.name):
            node.name = self.rename(node, node.name, "as")
        node.body = [self.visit(s) for s in node.body]
        return node

    def visit_MatchAs(self, node: ast.MatchAs) -> ast.AST:
        if node.pattern is not None:
            node.pattern = self.visit(node.pattern)
        if node.name and self.resolves_to_module(node.name):
            node.name = self.rename(
                node, node.name, "as" if node.pattern is not None else None)
        return node

    def visit_MatchStar(self, node: ast.MatchStar) -> ast.AST:
        if node.name and self.resolves_to_module(node.name):
            node.name = self.rename(node, node.name, "*")
        return node

    def visit_MatchMapping(self, node: ast.MatchMapping) -> ast.AST:
        self.generic_visit(node)
        if node.rest and self.resolves_to_module(node.rest):
            node.rest = self.rename(node, node.rest, "**")
        return node


def apply_edits(source: str, edits: "list[Edit]") -> str:
    """The source with each edit made at its token, and nothing else moved."""
    # ⛔ Not splitlines(): it also breaks at \x1c-\x1e, \x85 and \u2028,
    # which a string literal may hold, and the AST and the tokenizer count
    # only "\n".
    lines = io.StringIO(source).readlines()
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(row: int, byte_col: int) -> int:
        # The AST counts columns in UTF-8 bytes.
        line = lines[row - 1].encode()
        return starts[row - 1] + len(line[:byte_col].decode())

    tokens = [token for token in tokenize.generate_tokens(
        io.StringIO(source).readline)
        if token.type not in (tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT,
                              tokenize.INDENT, tokenize.DEDENT)]
    token_at = [(starts[t.start[0] - 1] + t.start[1], t) for t in tokens]

    spans: "list[tuple[int, int, str]]" = []
    for edit in edits:
        node = edit.node
        begin = offset(node.lineno, node.col_offset)  # type: ignore[attr-defined]
        end = offset(node.end_lineno, node.end_col_offset)  # type: ignore[attr-defined]
        if edit.whole:
            spans.append((begin, end, edit.new))
            continue
        previous = None
        found = False
        for at, token in token_at:
            if at >= end:
                break
            if (at >= begin and token.type == tokenize.NAME
                    and token.string == edit.old
                    and (edit.after is None or previous == edit.after)):
                spans.append((at, at + len(edit.old), edit.new))
                found = True
                # `global a, b` holds every name in one statement; any
                # other edit is one token.
                if not isinstance(node, ast.Global):
                    break
            previous = token.string
        if not found:
            raise ValueError(f"no `{edit.old}` token at line {node.lineno}")  # type: ignore[attr-defined]
    out = source
    for begin, end, text in sorted(set(spans), reverse=True):
        out = out[:begin] + text + out[end:]
    return out


# --- which cases can join -----------------------------------------------------

def declared_layers() -> "set[str]":
    declared: "set[str]" = set()
    for line in LAYERS.read_text().splitlines():
        fields = line.split("#", 1)[0].split()
        if fields:
            declared.add(fields[0])
    return declared


def excluded() -> "dict[str, str]":
    out: "dict[str, str]" = {}
    if EXCLUSIONS.exists():
        for line in EXCLUSIONS.read_text().splitlines():
            fields = line.split("#", 1)[0].split(None, 1)
            if fields:
                out[fields[0]] = fields[1] if len(fields) > 1 else ""
    return out


def refusal(case: pathlib.Path, layers: "set[str]",
            exclusions: "dict[str, str]") -> "str | None":
    """Why `case` cannot join a batch, or None."""
    if case.stem in exclusions:
        return f"excluded: {exclusions[case.stem]}"
    if f"{case.parent.name}/{case.stem}" in layers:
        return "it declares a layer"
    stdout = case.with_suffix(".stdout")
    if not stdout.exists():
        return "it has no .stdout"
    exitcode = case.with_suffix(".exitcode")
    if exitcode.exists() and exitcode.read_text().strip() != "0":
        return "it expects a non-zero exit"
    if case.with_suffix(".stderr-re").exists():
        return "it checks stderr"
    source = case.read_text()
    if MARKER in source:
        return "it spells the marker"
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "CPython does not parse it"
    # The code without its comments: a comment saying "exit code" is not a
    # call to exit().
    found = CONTEXT.search(ast.unparse(tree))
    if found:
        return f"it can observe its context (`{found.group(0)}`)"
    names = module_bindings(tree)
    if isinstance(names, str):
        return names
    for imported in module_imports(tree).values():
        top = imported.lstrip(".").split(".")[0]
        if imported.startswith(".") or (case.parent / f"{top}.py").exists():
            # The program is in another directory, where it cannot be found;
            # and the multi-module axis is what such a case is for.
            return f"it imports `{top}` from beside it"
    # A name the program could print would print renamed. A def's or a
    # class's name is in its repr, its __name__ and the messages that name
    # it, so one spelled as a word in the expected output -- or in a string
    # the program computes with, as `repr(x).startswith("<__main__.Plain")`
    # does, printing only True -- keeps the case out. A docstring is not
    # computed with and does not count.
    docstrings = {id(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Expr)}
    output = stdout.read_text()
    text = output + "\n".join(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and id(node) not in docstrings)
    declared = {node.name for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef))}
    for name in sorted(names & declared):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])",
                     text):
            return (f"its output or a string in it spells the name of "
                    f"`{name}`")
    # A variable's name reaches the output only through `f"{x=}"` or a
    # message that quotes it ("name 'x' is not defined").
    if any(isinstance(node, ast.JoinedStr) and any(
            isinstance(part, ast.Constant) and isinstance(part.value, str)
            and part.value.rstrip().endswith("=") for part in node.values)
           for node in ast.walk(tree)):
        return "it prints a variable by name (`f\"{x=}\"`)"
    for name in sorted(names - declared):
        if re.search(rf"['\"]{re.escape(name)}['\"]", output):
            return f"its output quotes the variable name `{name}`"
    try:
        joined_source([case])
    except (ValueError, tokenize.TokenError) as error:
        return f"it cannot be renamed: {error}"
    shadowed = names & set(dir(builtins))
    if shadowed:
        # Renamed, a rebinding of `print` would stop shadowing for the
        # case's own functions only if the renamer missed a scope; refusing
        # costs a handful of cases and removes the question.
        return f"it rebinds the builtin `{sorted(shadowed)[0]}`"
    return None


def joined_source(cases: "list[pathlib.Path]") -> str:
    parts = []
    for index, case in enumerate(cases):
        source = case.read_text()
        tree = ast.parse(source)
        names = module_bindings(tree)
        assert isinstance(names, set)
        renamer = Renamer(names, f"_c{index}_")
        renamer.visit(tree)
        renamed = apply_edits(source, renamer.edits)
        # The renamer renamed its tree as well; the edited source has to
        # parse back to exactly that tree, or an edit landed on the wrong
        # token.
        if ast.dump(ast.parse(renamed)) != ast.dump(tree):
            raise ValueError(f"{case.name}: the renamed source does not "
                             f"parse to the renamed tree")
        if not renamed.endswith("\n"):
            renamed += "\n"
        parts.append(f"# --- {case.name}\n{renamed}print({MARKER!r})\n")
    return "\n".join(parts)


# --- running ------------------------------------------------------------------

def compatible_groups(cases: "list[pathlib.Path]") -> "list[list[pathlib.Path]]":
    """The cases in order, cut wherever one imports a name an earlier case in
    the same group imported from somewhere else."""
    groups: "list[list[pathlib.Path]]" = []
    seen: "dict[str, str]" = {}
    for case in cases:
        imports = module_imports(ast.parse(case.read_text()))
        if not groups or any(seen.get(name, source) != source
                             for name, source in imports.items()):
            groups.append([])
            seen = {}
        groups[-1].append(case)
        seen.update(imports)
    return groups


class Outcome:
    def __init__(self) -> None:
        self.passed: "list[str]" = []
        self.failed: "list[pathlib.Path]" = []
        self.disagreed: "list[str]" = []
        self.alone: "list[str]" = []


def run_alone(lyc: pathlib.Path, case: pathlib.Path, timeout: float) -> bool:
    result = subprocess.run(
        [sys.executable, str(RUN_CASE), "--lyc", str(lyc),
         "--timeout", str(timeout), str(case)],
        capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        sys.stdout.write(result.stderr)
    return result.returncode == 0


def run_batch(lyc: pathlib.Path, cases: "list[pathlib.Path]", timeout: float,
              work: pathlib.Path, outcome: Outcome, keep: "pathlib.Path | None",
              depth: int = 0) -> None:
    source = work / f"batch_{depth}_{cases[0].stem}.py"
    source.write_text(joined_source(cases))
    if keep is not None:
        (keep / source.name).write_text(source.read_text())
    try:
        result = subprocess.run(
            [str(lyc), "jit", str(source)], capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=timeout * len(cases))
        stdout, code = result.stdout, result.returncode
    except subprocess.TimeoutExpired:
        stdout, code = "", None
    slices = stdout.split(MARKER + "\n")
    finished = len(slices) - 1
    if finished == 0 and code != 0 and len(cases) > 1:
        # Nothing ran: most likely it did not compile. Halve it.
        middle = len(cases) // 2
        run_batch(lyc, cases[:middle], timeout, work, outcome, keep, depth + 1)
        run_batch(lyc, cases[middle:], timeout, work, outcome, keep, depth + 1)
        return
    for index, case in enumerate(cases):
        name = f"{case.parent.name}/{case.stem}"
        if index < finished and slices[index] == case.with_suffix(".stdout").read_text():
            outcome.passed.append(name)
            continue
        outcome.alone.append(name)
        alone = run_alone(lyc, case, timeout)
        if not alone:
            outcome.failed.append(case)
        elif index < finished:
            outcome.disagreed.append(name)
            print(f"BATCH-DISAGREE {name}: passes alone, but printed this "
                  f"in the batch:\n{slices[index]}", flush=True)
        # Stopped part way and passes alone: the case after the last
        # finished marker is the one the batch died in, or a later one
        # that never ran -- neither says anything about this case.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lyc", type=pathlib.Path)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--list-eligible", action="store_true",
                    help="print the cases that can join a batch, ';'-joined")
    ap.add_argument("--why", action="store_true",
                    help="print each case that cannot join, and why")
    ap.add_argument("--keep", type=pathlib.Path, default=None,
                    help="write each joined program into this directory")
    ap.add_argument("cases", nargs="*", type=pathlib.Path)
    args = ap.parse_args()

    layers = declared_layers()
    exclusions = excluded()
    if args.list_eligible or args.why:
        eligible = []
        for case in args.cases:
            why = refusal(case, layers, exclusions)
            if why is None:
                eligible.append(str(case))
            elif args.why:
                print(f"{case.stem}: {why}")
        if args.list_eligible:
            print(";".join(eligible), end="")
        return 0

    if args.lyc is None:
        print("--lyc is required", file=sys.stderr)
        return 2
    lyc = args.lyc.resolve()
    cases = [case.resolve() for case in args.cases]
    for case in cases:
        why = refusal(case, layers, exclusions)
        if why is not None:
            print(f"{case.name} cannot join a batch: {why}", file=sys.stderr)
            return 2
    if args.keep is not None:
        args.keep.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    outcome = Outcome()
    with tempfile.TemporaryDirectory() as scratch:
        for group in compatible_groups(cases):
            run_batch(lyc, group, args.timeout, pathlib.Path(scratch), outcome,
                      args.keep)
    print(f"{len(outcome.passed)} passed in the batch, {len(outcome.alone)} "
          f"ran alone, {len(outcome.failed)} failed, {len(outcome.disagreed)} "
          f"disagreed ({time.monotonic() - started:.1f} s)")
    for case in outcome.failed:
        print(f"FAILED {case.parent.name}/{case.stem} (alone: {RUN_CASE} "
              f"--lyc {lyc} {case})")
    return 1 if outcome.failed or outcome.disagreed else 0


if __name__ == "__main__":
    sys.exit(main())
