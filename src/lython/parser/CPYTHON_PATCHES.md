# CPython 3.14 Parser Patches

Lython vendors a reduced CPython 3.14 parser surface and patches it so the
generated parser can be compiled and linked without CPython runtime internals.

Vendored files:

- `parser.c`: copied from CPython `Parser/parser.c` and compiled into
  `LythonParser`.
- `python.gram`: copied from CPython `Grammar/python.gram` and used as the PEG
  contract input.
- `Tokens`: copied from CPython `Grammar/Tokens` and used as the token
  vocabulary contract.
- `Python.asdl`: copied from CPython `Parser/Python.asdl` and used as the AST
  shape contract.
- `unicodename_db.h`: copied from CPython `Modules/unicodename_db.h` for local
  `\N{...}` escape decoding.

Local patches:

- `pegen.h` shadows CPython `Parser/pegen.h` with a Lython-owned ABI. It keeps
  the generated parser's type surface while replacing CPython object, arena,
  token, and AST handles with opaque Lython-compatible handles.
- `PegenShim.c` provides the pegen/action helper ABI needed to link the
  generated parser. It implements the token-buffer fill contract, packrat memo
  table used by left-recursive rules, soft-keyword matching, and a small parse
  arena for temporary ASDL/action records. It also records the generated
  parser's returned root kind, direct child kind digest, and direct child shape
  digest so C++ can verify that file, single, eval, and function-type parse
  modes do not cross root semantics. ASDL sequences are registered while parsing so
  `_PyPegen_seq_flatten` can preserve CPython's statement-list semantics
  without guessing arbitrary pointers.
- `Parser.cpp` feeds the Lython lexer token stream into
  `lython_cpython_generated_parse_tokens()` before C++ AST emission. Inputs
  rejected by CPython's generated PEG parser are rejected before the
  hand-written AST builder runs. After public AST emission, the root shape,
  direct child kind digest, and direct child shape digest are checked against
  the generated parser's root handle, including `FunctionType.argtypes` plus
  `returns`. CI checks for `_PyPegen_parse`.

The generated parser source is therefore patched and linked, but CPython's
runtime, allocator, `PyObject` implementation, and C API are not linked.

Current bridge constraints:

- The public `Node` tree is still emitted by Lython's C++ AST builder after the
  generated PEG parser accepts the token stream. The `_PyAST_*` actions in
  `parser.c` are represented only enough for parser execution and must still be
  connected to full `Node` construction before the hand-written AST builder can
  be removed.
- Lython's current lexer still keeps f-strings and t-strings as whole tokens,
  but the generated-parser bridge now expands them into CPython-style
  `FSTRING_*`/`TSTRING_*` token streams before calling the patched parser. The
  existing C++ AST builder remains responsible for public
  `JoinedStr`/`TemplateStr` node construction.
- `_PyPegen_concatenate_strings` and `_PyPegen_concatenate_tstrings` preserve
  enough root shape for validation. Plain adjacent string literals remain
  `Constant`, while f-string/t-string paths keep `JoinedStr`/`TemplateStr`
  rather than collapsing to a generic constant stub.
- Many `_PyAST_*` action shims now preserve recursive child structure rather
  than kind-only stubs. Validation normalizes away CPython enum-only children
  such as `operator`, `cmpop`, and `expr_context`; literal `Constant` chunks
  directly under `JoinedStr`/`TemplateStr` are ignored only for root-level child
  shape comparison and kept by the recursive AST shape check.
