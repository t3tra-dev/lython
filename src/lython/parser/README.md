# Lython Parser

This module is the C++ frontend bootstrap that replaces the previous
`ast.parse`-based Python frontend.

Vendored CPython 3.14 parser files live directly in this directory and are
treated as the grammar/token/ASDL authority. The generated `parser.c` is
compiled into `LythonParser` through the patched `pegen.h` and
`PegenShim.c` ABI instead of CPython's runtime ABI. Lython exposes its
own C++ AST model and parser API rather than linking CPython object, arena, or
tokenizer internals. Patch details are recorded in `CPYTHON_PATCHES.md`.

Imported from `https://github.com/python/cpython/tree/3.14`:

- `python.gram` from `Grammar/python.gram`
- `Tokens` from `Grammar/Tokens`
- `parser.c` from `Parser/parser.c`
- `Python.asdl` from `Parser/Python.asdl`
- `unicodename_db.h` from `Modules/unicodename_db.h`
- `CPython-3.14.LICENSE` from `LICENSE`
- `pegen.h` is a Lython patch that shadows CPython `Parser/pegen.h`.
- `PegenShim.c` is a Lython patch that supplies the pegen/action helper
  ABI without CPython runtime linkage.

Current C++ entry points:

- `lython::parser::cpython314Spec()` loads the vendored grammar, tokens, and
  ASDL spec. It also loads the generated `parser.c` as contract source text and
  indexes the CPython-generated `*_rule(Parser *p)` declarations. The same
  `parser.c` is compiled and linked through the patched pegen ABI, so drift is
  caught by startup contract validation, C compilation, and the runtime
  generated-parser gate in `parse()`.
- `lython::parser::parseCpythonPegGrammar()` parses `python.gram` into
  a small Lython PEG IR. It keeps rule names, return types, memo markers, and
  FIRST sets. The public PEG IR now also preserves the expression tree for each
  rule and top-level alternative, including CPython labels such as
  `n=NAME`, `params=[params]`, or `b=block`. Top-level rule alternatives also
  retain their CPython action bodies as contract text, their called
  action/helper symbols, the `_PyAST_*` constructor names referenced by those
  calls, their head rule, and per-alternative FIRST sets plus referenced
  rules, tokens, and literals. The generated C parser is compiled through
  `pegen.h`/`PegenShim.c`; CPython action calls are therefore routed to
  Lython-owned ABI hooks instead of CPython `Python.h`, arena allocation, or
  `_PyAST_*` runtime dependencies.
  Before the hand-written C++ AST builder emits a public `Node` tree,
  `parse()` feeds the Lython lexer token stream to the compiled CPython
  generated parser through `lython_cpython_generated_parse_tokens()`. This
  exercises CPython's PEG control flow, including left-recursive packrat memo,
  soft-keyword matching, and EOF token filling, without linking CPython's
  runtime. The bridge records the generated parser's returned root kind, root
  shape, direct child kind digest, direct child shape digest, and recursive AST
  shape digest, then checks those against the requested parse mode and emitted
  public AST root.
  `FunctionType` validation includes both `argtypes` and `returns`.
  Whole-token f-strings and t-strings from the Lython lexer are split by the
  generated-parser bridge into CPython-style
  `FSTRING_*`/`TSTRING_*` token streams before validation; the C++ AST builder
  continues to emit the public `JoinedStr`/`TemplateStr` nodes.
  `CpythonPegAdapter` owns FIRST-set, alternative-head matching, statement
  form selection, atom/primary form classification, comparison-operator
  classification, arithmetic/bitwise/unary operator family classification,
  lambda/assignment-expression/await/comprehension clause starts, and
  rule-to-AST-constructor/action-helper lookup for the recursive-descent
  bootstrap. It also owns CPython rule literal matching used by
  simple-statement keyword consumption. Token-level helpers such as operator
  longest-prefix matching are exposed through `CpythonSpec` APIs rather than
  reading spec internals from the lexer. Parser dispatch, emitted AST kind
  selection, and token acceptance are therefore derived from the vendored PEG
  and token metadata while Lython keeps its own `Node` builder. Parser startup
  validates that expression precedence, primary, atom, lambda, and
  function-type entry rules still match the vendored CPython 3.14 PEG shape.
  FIRST sets are split into the normal CPython first-pass view, which excludes
  `invalid_*` rules, and a recovery view that includes those diagnostic rules
  for future second-pass error reporting.
- `lython::parser::parse()` returns a Python `ast`-shaped generic AST.
- `lyc parse <file.py>` dumps the C++ AST without invoking CPython.
- `lyc parse --type-comments <file.py>` mirrors `ast.parse(...,
  type_comments=True)` for `# type: ignore` and supported statement
  `type_comment` fields. CPython 3.14's `with_stmt` shape is preserved: sync
  `with` and unparenthesized `async with` can carry a statement type comment,
  while parenthesized `async with (...)` leaves `type_comment=None`.
- `lyc parse --interactive <file.py>` mirrors `ast.parse(..., mode="single")`
  and returns an `Interactive` root for a single interactive statement. It
  follows the vendored CPython 3.14 `statement_newline` rule, including
  `NEWLINE` as `Pass()`.
- `lyc parse --expression <file.py>` mirrors `ast.parse(..., mode="eval")`
  using the vendored CPython 3.14 `expressions` rule, including top-level
  tuple expressions such as `1, 2`, and rejects trailing tokens after the
  expression.
- `lyc parse --function-type <file.py>` mirrors `ast.parse(...,
  mode="func_type")` for function type comments, including the vendored
  CPython 3.14 `type_expressions` rule where `*` and `**` are accepted but
  ignored in the resulting `FunctionType.argtypes`.

The C++ parser is no longer treated as a private Lython grammar. It derives
hard/soft keywords from the CPython-generated `parser.c` keyword tables,
operator spellings from `Tokens`, and
accepted AST node kinds, ASDL enum alternatives, and AST field schemas from
`Python.asdl`. The keyword tables inferred from `python.gram`
are retained as a startup consistency check against generated `parser.c`, not
as the lexer's source of truth. Statement dispatch goes through
`CpythonPegAdapter`, which matches parsed PEG alternative heads by CPython
FIRST sets rather than line-oriented grammar text scraping or private Lython
keyword tables. Startup contract validation also checks that the CPython 3.14 PEG
still exposes the statement, function/class, parameter, call-argument,
expression, comprehension, and pattern rule structure that the hand-written
recursive-descent code implements. If the vendored grammar or generated parser
drifts, parsing fails before emission instead of silently accepting a stale
Lython-specific grammar. The lexer stores the exact CPython token name and
numeric token id derived from `Tokens` or the generated `parser.c`
hard-keyword table for each token, e.g. `LPAR`, `RARROW`, `ELLIPSIS`, `NAME`,
`NUMBER`, `STRING`, `FSTRING_START`, and `TSTRING_START`. Parser rule-entry
checks compare both literal FIRST entries and these CPython token names. Hard
keywords keep their CPython `NAME` token identity for AST/name handling but
also retain the generated parser keyword token id needed by future table-driven
dispatch; they are excluded from generic `NAME` FIRST matches unless the
grammar literal itself matched first. Soft keywords remain valid names outside
their contextual grammar rules.
`parse()` rejects startup if required CPython 3.14
entry rules such as `file`, `eval`, `statement`, `simple_stmt`, or
`compound_stmt` are absent. Parse output is checked against ASDL node kinds,
field names, field value categories, optional fields, sequence/nullability
markers, ASDL field order, and parse-mode root kind (`Module`,
`Interactive`, `Expression`, or `FunctionType`) so parser drift is caught
before lowering.
Startup validation additionally rejects any PEG FIRST-set token that is absent
from the vendored `Tokens`, rejects operator spellings without a
corresponding named CPython token, and rejects disagreement between
`python.gram` keyword literals and the generated `parser.c` keyword arrays.
The contract also reads generated `*_type` rule IDs and hard-keyword token IDs
from `parser.c` and rejects missing or duplicate entries. It also reads
generated literal-token references such as
`_PyPegen_expect_token(p, 7)  // token='('` from `parser.c` and verifies that
their numeric IDs match the ordinal token IDs derived from `Tokens`.
This keeps the
hand-written AST builder bound to the CPython 3.14 token vocabulary and
generated parser acceptance while avoiding a direct dependency on CPython's
runtime object model or arena allocator. The generated `parser.c` ABI is
patched locally through `pegen.h` and `PegenShim.c`.
Representative PEG action references such as `_PyPegen_make_module`,
`_PyAST_Return`, and `_PyAST_Raise` are also checked at startup, so changes in
CPython's AST-construction intent are surfaced before Lython lowers stale
hand-written semantics. Every `_PyAST_*` constructor referenced by an action is
also checked against the vendored `Python.asdl`, keeping grammar actions and
ASDL node families in the same contract. The startup contract additionally
checks that the CPython-generated `parser.c` still exposes generated rule
functions for the PEG rules Lython relies on, and that representative AST
actions still expose the grammar labels used to build their nodes. Entry
rules such as `file`, `interactive`, `eval`, `func_type`, and
`statement_newline` are tied to their CPython helper/action boundaries
(`_PyPegen_make_module`, `_PyAST_Interactive`, `_PyAST_Expression`,
`_PyAST_FunctionType`, and the interactive blank-line `_PyAST_Pass`). Simple
statement rules with a single
AST constructor, such as `return_stmt`, `raise_stmt`, `pass_stmt`,
`break_stmt`, `continue_stmt`, `global_stmt`, `nonlocal_stmt`, `del_stmt`,
`yield_stmt`, and `assert_stmt`, are exposed through the PEG rule metadata and
used by the bootstrap parser when choosing the emitted ASDL node kind.
Compound statement nodes are being moved through the same path:
`function_def_raw`, `class_def_raw`, `if_stmt`/`elif_stmt`, `while_stmt`,
`for_stmt`, `with_stmt`, `try_stmt`, and `match_stmt` now expose and validate
their expected AST constructor set before the bootstrap parser emits those
node kinds. The same metadata now drives expression and pattern constructors
that CPython spells as `_PyAST_*` actions, including `NamedExpr`, `Lambda`,
`Yield`/`YieldFrom`, `IfExp`, `BoolOp`, `Compare`, arithmetic
`BinOp`/`UnaryOp`, `Await`, `Attribute`/`Call`/`Subscript`, `Slice`,
load-context `Tuple`/`List`/`Dict`/`Set`, comprehensions, `GeneratorExp`, and
`Match*` pattern nodes. Lowercase ASDL nodes that are also produced by direct
CPython actions, such as `arg`, `alias`, `keyword`, `withitem`,
`ExceptHandler`, `match_case`, and `comprehension`, use the same route.
The contract also checks the CPython 3.14 f-string and t-string middle,
replacement-field, conversion, and format-spec rule shape, including the
`_PyPegen_joined_str`, `_PyPegen_template_str`,
`_PyPegen_formatted_value`, `_PyPegen_interpolation`, and
`_PyPegen_concatenate_strings`/`_PyPegen_concatenate_tstrings`
action-helper boundary that the C++ parser mirrors when emitting `JoinedStr`,
`FormattedValue`, `TemplateStr`, and `Interpolation` nodes. The patched
generated-parser shim preserves root shape through these helpers: plain
adjacent strings validate as `Constant`, while f-string/t-string expressions
validate as `JoinedStr`/`TemplateStr`.
The root child shape digest validates each root-level statement/expression as
`Kind(childKinds...)`. The recursive shape digest then checks every structural
ASDL child in the generated action tree against the emitted public AST tree.
Both checks treat CPython enum-only AST families (`operator`, `unaryop`,
`boolop`, `cmpop`, and `expr_context`) as scalar tags rather than structural
children. Literal `Constant` chunks directly under `JoinedStr`/`TemplateStr`
are ignored only in the root-level child digest; recursive validation keeps
them, including debug f-string/t-string text.
CPython token-returning paths such as bare `NAME` are kept as explicit Lython
AST construction because they are not `_PyAST_*` actions in `python.gram`.
Operator tokens are likewise limited to spellings derived from
`Tokens`; unknown punctuation such as `$` is rejected by the lexer
instead of being forwarded as a private Lython operator token.
`parse()` also validates that the vendored `statement`, `simple_stmt`, and
`compound_stmt` rules still expose the expected CPython 3.14 alternative heads
such as `compound_stmt`, `simple_stmts`, `return_stmt`, `import_stmt`,
`function_def`, `class_def`, `match_stmt`, and loop/exception statement rules.
If the vendored grammar changes shape, Lython fails at parser startup rather
than silently falling back to stale dispatch logic.
The parser also performs CPython action-helper-style validation for parameter
lists, type parameter lists, assignment/delete targets, grammar-level pattern
forms, call argument ordering, `with` item separators, and `try` handler
ordering, so invalid constructs such
as empty type parameter lists, duplicate `/`, `/` after `*`, bare `*` without
keyword-only parameters, missing comma between `/` and `*`, arguments after
`**kwargs`, duplicate `*args`, bare `*` with an associated type comment,
missing default values, defaults on
`*args`/`**kwargs`, assignment to non-target expressions, invalid `as` pattern
targets, tuple/list annotated assignment targets, parenthesized lambda
parameters, standalone star patterns
such as `case *rest:` outside sequence patterns, positional call arguments
after keyword arguments, iterable unpacking after keyword unpacking,
assignment to call argument unpacking such as `*args=...` or `**kwargs=...`,
invalid keyword targets such as `True=...` or `obj.attr=...`,
unparenthesized generator expressions after another argument,
keyword assignments followed by generator-expression clauses,
invalid delete targets such as function calls, literals, starred expressions,
and comprehensions,
invalid assignment and augmented-assignment targets such as `yield = ...`,
function calls, literals, and tuple augmented assignment,
invalid named-expression targets and mistaken `=` in named-expression
contexts,
conditional expressions missing `else` or the expression after `else`,
missing commas between list or tuple/grouped expression elements,
unparenthesized `not` after arithmetic or unary operators,
Python 2-style `print x` and `exec x` statements,
decorators applied to non-function/class statements,
multiple function type comments on a single `def`,
non-parenthesized trailing commas in `with`, invalid `for` targets, missing
`in` in `for` statements, missing indented suite
blocks, empty `import` targets, misplaced `import ... from ...`, invalid import
aliases, extra targets after `from ... import *`, bare `except*`, bare `except:`
before another handler, and `try`
blocks without `except` or `finally` are rejected before emission. Dict/set
literal invalid forms such as missing `:` after dictionary keys, missing commas
between set elements, missing values after `:`, starred dictionary values, and
assigning to dictionary unpacking are also rejected in the parser. CPython 3.14
invalid `def`/`class`/`if`/`elif`/`else`/`for`/`with`/`try`/`while` and
`match`/`case` forms such as missing `:`, missing indented suite blocks,
standalone `elif`/`else`, `elif` after `else`, missing `try` handlers, and
mixed `except`/`except*` handlers produce dedicated diagnostics instead of
falling through to expression parsing. `except`/`except*` clauses also reject
missing suite blocks and non-name `as` targets. Python 3.14
unparenthesized multiple exception types such as `except A, B:` and
`except* A, B:` are emitted as `ExceptHandler(type=Tuple(..., ctx=Load))`
when no `as` alias is present; `as` aliases still require a single exception
expression. Standalone `except`/`finally`
clauses and bare `except`/`except*` headers synchronize as invalid clauses
instead of cascading through expression parsing.
CPython 3.14
invalid comprehension forms such as starred comprehension elements, missing
`in` after comprehension targets, and unparenthesized tuple targets before a
comprehension clause produce dedicated parser diagnostics.
Invalid grouped expressions such as `(*x)` and `(**x)` are rejected while
starred tuple expressions such as `(*x,)` remain valid.
Assignment expressions follow the CPython `named_expression` boundary:
unparenthesized `:=` is accepted in named-expression contexts such as
`if`/`while` tests, match guards and subjects, positional call arguments,
subscripts, parenthesized groups, and list/tuple/set displays, but rejected
from plain expression contexts such as expression statements, lambda bodies,
and keyword argument values.
Starred operands follow CPython's two grammar families: display/expression-list
`star_expression` parses only `*bitwise_or`, while call arguments and
multi-element subscript slices use `starred_expression` and therefore allow
`*expression`.
`for` and `async for` iterable expressions follow CPython's `for_stmt`
`star_expressions` boundary, so comma-separated and starred tuple iterables
such as `for x in a, b:` or `for x in *a, b:` are emitted as
`Tuple(..., ctx=Load)` while comprehension iterables keep their narrower
`disjunction` grammar.
Comma-separated expression statements such as `1, 2, 3` and `*x,` are emitted
as `Expr(Tuple(..., ctx=Load))` unless a following assignment operator turns
the tuple into an assignment target. This mirrors CPython's `star_expressions`
statement boundary rather than treating every comma-leading statement as a
tuple assignment.
Direct `yield` expressions follow CPython's `yield_expr` boundary: standalone
`yield` statements and assignment/augmented-assignment annotated RHS values are
accepted, parenthesized `yield` remains valid in expression positions, and
unparenthesized `yield` is rejected from ordinary expressions such as
`return yield 1`, call arguments, list displays, and condition tests.
The `with` parser also preserves CPython's ambiguity rule for parenthesized
items: forms such as `with (a, b):` are parsed as parenthesized `with_item`
lists, while `with (x := cm):` and `with (x := cm), y:` fall back to grouped
context expressions because top-level `:=` is not valid inside `with_item`
lists.
Pattern binding consistency follows CPython's phase split: the parser emits the
AST for duplicate captures, OR-pattern alternatives with different capture
sets, duplicate literal mapping keys, duplicate class-pattern keyword
attributes, and multiple starred names inside a sequence pattern. Those cases
are compile/verification errors rather than `ast.parse` errors. Duplicate
function/lambda arguments and duplicate type parameter names follow the same
phase split. Mapping patterns still reject bare-name keys before AST emission
while allowing repeated dotted value-pattern keys.
`invalid_type_param` cases such as `*Ts: int` and `**P: int` also produce
dedicated parser diagnostics rather than cascading delimiter errors. Starred
subscript expressions such as `tuple[*Ts]`, and Python 3.14 `TypeVarTuple`
defaults such as `type A[*Ts = *tuple[int]] = tuple[*Ts]`, are represented
with the CPython `Starred(ctx=Load)` AST shape.
Function vararg annotations also follow CPython's `param_star_annotation`
boundary: `def f(*args: *Ts): ...` emits
`arguments.vararg.annotation=Starred(Name("Ts"), ctx=Load())`, while normal
parameters and `**kwargs` keep the ordinary `annotation -> expression` rule.
Expression contexts and operators are emitted as ASDL-style nodes such as
`Load()` and `Add()` instead of private string tags.
Numeric literal handling follows CPython integer token constraints for leading
zero decimal literals, base-prefixed integers, underscores, imaginary suffixes,
and arbitrary precision integer AST values. Integers that fit in `int64_t` stay
on the compact AST path; wider values are preserved as decimal `BigInteger`
constants and emitted as `py.int.constant` values.
String literal handling includes CPython-style adjacent string/f-string
concatenation and Python 3.14 `TemplateStr`/`Interpolation` AST construction
for t-strings; mixed t-string and string/f-string/bytes literal concatenation
is rejected before AST emission. Adjacent `Constant(str)` parts inside
`JoinedStr` and `TemplateStr` are folded and empty constant chunks are dropped,
mirroring CPython's `_PyPegen_joined_str`/`_build_concatenated_str` behavior.
F-string and t-string replacement fields are found from the raw literal
spelling, so escaped braces such as `\x7b` become literal text rather than
replacement-field delimiters. Replacement fields handle PEP 701-style
same-quote string literals, comments/newlines inside fields, nested format
specifications, debug `=` forms, and CPython-style invalid conversion or empty
expression diagnostics, including the unparenthesized lambda rule inside
replacement fields. Python 3.14 `Interpolation.str` keeps the source
expression spelling with leading whitespace preserved and only trailing
whitespace/`=` stripped, matching CPython's `_strip_interpolation_expr`
boundary. Doubled braces remain literal text and single `}` braces are
rejected. Non-raw string and bytes literals decode the
common CPython escapes (`\n`, `\t`, `\xNN`, octal escapes, and `\u`/`\U`
Unicode scalar escapes for strings) before the AST is built; invalid strict
`\x`, `\u`, and `\U` escapes are rejected. Bytes literals reject unescaped
non-ASCII source characters while still allowing escaped byte values. Raw
strings keep the source spelling. `\N{...}` named Unicode escapes are decoded
through a local C++ lookup over CPython 3.14's vendored
`unicodename_db.h` tables. The parser supports the packed Unicode name
DAWG, name aliases such as `LF`, `HT`, `NBSP`, `ZWJ`, and `BOM`, and generated
name ranges for Hangul syllables, CJK unified ideographs, and Tangut
ideographs. Unicode named sequences are expanded into their UTF-8 codepoint
sequence. None of these paths link against CPython internals.
Single-quoted string literals reject unescaped physical newlines as tokenizer
errors, while backslash line continuations in both LF and CRLF form remain
valid. Source text is decoded before tokenization: UTF-8 and UTF-8 BOM inputs
are validated as strict UTF-8, and NUL bytes are rejected. A UTF-8 BOM at the
start of a byte buffer is consumed before normal lexing, matching CPython's
tokenizer boundary for UTF-8 source input. PEP 263 coding cookies are checked
on the first two physical lines using the vendored CPython tokenizer rule:
`utf-8` continues on the normal path, common ASCII spellings that CPython
accepts through its codec registry (`ascii`, `us-ascii`, `iso646-us`, and
`ansi_x3.4_1968`-style aliases) are validated as ASCII, Latin-1 spellings
normalized by CPython's tokenizer helper (`latin-1`, `iso-8859-1`, and
`iso-latin-1`) are decoded to UTF-8 locally, and unsupported codecs are
rejected without linking CPython's codec registry. BOM plus non-UTF-8 cookie
is rejected as an encoding problem.
`...` is represented as `Constant(value=Ellipsis)`, matching the
CPython ASDL-level constant shape used by annotations such as `tuple[T, ...]`.
Imaginary numeric literals are represented as
`Constant(value=std::complex<double>)`; complex lowering is intentionally a
later emitter/runtime feature. When requested via `ParseOptions::typeComments`,
CPython-style `TYPE_IGNORE` comments are emitted as `TypeIgnore` nodes and
simple statement/function type comments are attached to the ASDL
`type_comment` fields. `TypeIgnore.tag` includes CPython's trailing newline
payload even when the source file ends without a physical newline. Identifier
tokenization uses generated range tables derived from CPython 3.14's
`Objects/unicodetype_db.h` `XID_Start` and `XID_Continue` flags, including
`Other_ID_Start`/`Other_ID_Continue` compatibility characters such as `℘`,
`℮`, `ᢅ`, `ᢆ`, `·`, and Ethiopic digit continuations. Source such as `値 = 1`
therefore stays on the C++ parser path while obvious symbols such as emoji are
rejected. Identifier AST strings are normalized with a generated NFKC kernel
derived from CPython 3.14's `Modules/unicodedata_db.h` tables: NFKD
decomposition, canonical combining ordering, Hangul composition, and NFC
composition pairs are implemented locally instead of calling CPython
`unicodedata.normalize`. Raw token spelling is still retained for PEG
literal/keyword matching, so a fullwidth spelling that normalizes to `if`
remains a `Name(id="if")`, matching CPython's tokenizer/parser boundary.
Line-start form feed characters follow the CPython tokenizer rule and reset the
current indentation column. Indentation also tracks CPython's alternate
`ALTTABSIZE=1` column so ambiguous tab/space mixtures are rejected before PEG
parsing. Explicit line continuations that occur while scanning indentation also
follow CPython's `cont_line_col` rule: the first backslash before the continued
physical line fixes the logical indentation column, so the whitespace on the
continued line is not allowed to redefine the block indent.
The vendored PEG/ASDL files are the compatibility contract for the C++ parser:
startup validation rejects grammar, token, generated-rule, action-helper, and
ASDL drift before lowering. The public AST API remains the Lython `Node` model,
and any future generated or table-driven parser must target those `Node`
builders directly rather than compiling the vendored `parser.c` against
CPython runtime internals.
