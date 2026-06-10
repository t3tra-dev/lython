// Patched CPython 3.14 pegen interface for Lython.
//
// This header intentionally shadows CPython's Parser/pegen.h so the vendored
// generated parser.c can be compiled without CPython headers or runtime
// internals. The generated recursive-descent parser remains CPython's parser;
// semantic actions are routed through opaque handles that are implemented by
// Lython's C++ AST layer.
#ifndef LYTHON_CPYTHON_314_PATCHED_PEGEN_H
#define LYTHON_CPYTHON_314_PATCHED_PEGEN_H

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef ptrdiff_t Py_ssize_t;

typedef struct LythonPegenObject {
  int tag;
} PyObject;

typedef struct LythonPegenArena {
  void **allocations;
  size_t size;
  size_t capacity;
} PyArena;

typedef struct LythonPegenCompilerFlags {
  int cf_flags;
  int cf_feature_version;
} PyCompilerFlags;

typedef struct asdl_seq asdl_seq;

typedef void *identifier;
typedef void *string;
typedef void *constant;
typedef void *mod_ty;
typedef void *stmt_ty;
typedef void *comprehension_ty;
typedef void *excepthandler_ty;
typedef void *arguments_ty;
typedef void *arg_ty;
typedef void *keyword_ty;
typedef void *alias_ty;
typedef void *withitem_ty;
typedef void *match_case_ty;
typedef void *pattern_ty;
typedef void *type_param_ty;

typedef enum { Name_kind = 1, Call_kind = 2, Tuple_kind = 3 } expr_kind_ty;

typedef struct _expr {
  expr_kind_ty kind;
  int lineno;
  int col_offset;
  int end_lineno;
  int end_col_offset;
  union {
    struct {
      identifier id;
    } Name;
    struct {
      asdl_seq *args;
      asdl_seq *keywords;
    } Call;
  } v;
  const char *lython_kind;
  asdl_seq *lython_seq;
  void *lython_primary;
  void *lython_secondary;
} *expr_ty;

typedef int expr_context_ty;
typedef int boolop_ty;
typedef int operator_ty;
typedef int unaryop_ty;
typedef int cmpop_ty;

enum {
  Load = 1,
  Store,
  Del,
  And,
  Or,
  Add,
  Sub,
  Mult,
  MatMult,
  Div,
  Mod,
  Pow,
  LShift,
  RShift,
  BitOr,
  BitXor,
  BitAnd,
  FloorDiv,
  Invert,
  Not,
  UAdd,
  USub,
  Eq,
  NotEq,
  Lt,
  LtE,
  Gt,
  GtE,
  Is,
  IsNot,
  In,
  NotIn
};

typedef struct asdl_seq {
  Py_ssize_t size;
  void **elements;
} asdl_seq;

typedef asdl_seq asdl_stmt_seq;
typedef asdl_seq asdl_expr_seq;
typedef asdl_seq asdl_comprehension_seq;
typedef asdl_seq asdl_excepthandler_seq;
typedef asdl_seq asdl_arguments_seq;
typedef asdl_seq asdl_arg_seq;
typedef asdl_seq asdl_keyword_seq;
typedef asdl_seq asdl_alias_seq;
typedef asdl_seq asdl_withitem_seq;
typedef asdl_seq asdl_match_case_seq;
typedef asdl_seq asdl_pattern_seq;
typedef asdl_seq asdl_type_param_seq;
typedef asdl_seq asdl_identifier_seq;
typedef asdl_seq asdl_int_seq;

enum {
  ENDMARKER = 0,
  NAME = 1,
  NUMBER = 2,
  STRING = 3,
  NEWLINE = 4,
  INDENT = 5,
  DEDENT = 6,
  OP = 54,
  TYPE_IGNORE = 55,
  TYPE_COMMENT = 56,
  SOFT_KEYWORD = 57,
  FSTRING_START = 58,
  FSTRING_MIDDLE = 59,
  FSTRING_END = 60,
  TSTRING_START = 61,
  TSTRING_MIDDLE = 62,
  TSTRING_END = 63,
  COMMENT = 64,
  NL = 65,
  ERRORTOKEN = 66,
  ENCODING = 67
};

#define asdl_seq_LEN(seq) ((seq) ? (seq)->size : 0)
#define asdl_seq_GET(seq, index) ((seq)->elements[(index)])
#define asdl_seq_SET_UNTYPED(seq, index, value)                                \
  ((seq)->elements[(index)] = (void *)(value))

asdl_seq *asdl_generic_seq_new(Py_ssize_t size, PyArena *arena);
#define _Py_asdl_generic_seq_new asdl_generic_seq_new

#define Py_file_input 257
#define Py_single_input 256
#define Py_eval_input 258
#define Py_func_type_input 345

#define PyPARSE_DONT_IMPLY_DEDENT 0x0002
#define PyPARSE_IGNORE_COOKIE 0x0010
#define PyPARSE_BARRY_AS_BDFL 0x0020
#define PyPARSE_TYPE_COMMENTS 0x0040
#define PyPARSE_ALLOW_INCOMPLETE_INPUT 0x0100

#define CURRENT_POS (-5)
#define UNUSED(expr) ((void)(expr))

typedef struct _memo {
  int type;
  void *node;
  int mark;
  struct _memo *next;
} Memo;

typedef struct {
  int type;
  PyObject *bytes;
  int level;
  int lineno, col_offset, end_lineno, end_col_offset;
  Memo *memo;
  PyObject *metadata;
} Token;

typedef struct {
  const char *str;
  int type;
} KeywordToken;

typedef struct {
  struct {
    int lineno;
    char *comment;
  } *items;
  size_t size;
  size_t num_items;
} growable_comment_array;

typedef struct {
  int lineno;
  int col_offset;
  int end_lineno;
  int end_col_offset;
} location;

typedef struct {
  int type;
  const char *text;
  int lineno;
  int col_offset;
  int end_lineno;
  int end_col_offset;
} LythonCpythonToken;

typedef struct Parser {
  struct tok_state *tok;
  Token **tokens;
  int mark;
  int fill, size;
  PyArena *arena;
  KeywordToken **keywords;
  char **soft_keywords;
  int n_keyword_lists;
  int start_rule;
  int *errcode;
  int parsing_started;
  PyObject *normalize;
  int starting_lineno;
  int starting_col_offset;
  int error_indicator;
  int flags;
  int feature_version;
  growable_comment_array type_ignore_comments;
  Token *known_err_token;
  int level;
  int call_invalid_rules;
  int debug;
  location last_stmt_location;
} Parser;

typedef struct {
  cmpop_ty cmpop;
  expr_ty expr;
} CmpopExprPair;

typedef struct {
  expr_ty key;
  expr_ty value;
} KeyValuePair;

typedef struct {
  expr_ty key;
  pattern_ty pattern;
} KeyPatternPair;

typedef struct {
  arg_ty arg;
  expr_ty value;
  Token *tc;
} NameDefaultPair;

typedef struct {
  asdl_arg_seq *plain_names;
  asdl_seq *names_with_defaults;
} SlashWithDefault;

typedef struct {
  arg_ty vararg;
  asdl_seq *kwonlyargs;
  arg_ty kwarg;
} StarEtc;

typedef struct {
  operator_ty kind;
} AugOperator;

typedef struct {
  void *element;
  int is_keyword;
} KeywordOrStarred;

typedef struct {
  expr_ty value;
  PyObject *metadata;
} ResultTokenWithMetadata;

typedef enum {
  STAR_TARGETS = 0,
  DEL_TARGETS = 1,
  FOR_TARGETS = 2
} TARGETS_TYPE;

#define Py_None ((PyObject *)0)
#define Py_True ((PyObject *)1)
#define Py_False ((PyObject *)2)
#define Py_Ellipsis ((PyObject *)3)
#define PyExc_SyntaxError ((PyObject *)4)
#define PyExc_IndentationError ((PyObject *)5)
#define PyExc_IncompleteInputError ((PyObject *)6)

#define PyBytes_AS_STRING(obj) ((char *)(obj))
#define PyThreadState_Get() NULL
#define _PyThreadState_GET() NULL
#define _PyInterpreterState_GET() NULL
#define _Py_ReachedRecursionLimitWithMargin(tstate, margin) 0
#define PyErr_Occurred() 0
#define PyErr_NoMemory() ((void)0)
#define PyMem_Malloc malloc
#define PyMem_Realloc realloc
#define PyMem_Free free

#define CHECK(type, result) ((type)(result))
#define CHECK_NULL_ALLOWED(type, result) ((type)(result))
#define CHECK_VERSION(type, version, msg, node) ((type)(node))
#define NEW_TYPE_COMMENT(p, tc) NULL
#define EXTRA_EXPR(head, tail) 0, 0, 0, 0, NULL
#define EXTRA 0, 0, 0, 0, NULL
#define RAISE_SYNTAX_ERROR(msg, ...) NULL
#define RAISE_INDENTATION_ERROR(msg, ...) NULL
#define RAISE_SYNTAX_ERROR_ON_NEXT_TOKEN(msg, ...) NULL
#define RAISE_SYNTAX_ERROR_KNOWN_LOCATION(a, msg, ...) NULL
#define RAISE_SYNTAX_ERROR_KNOWN_RANGE(a, b, msg, ...) NULL
#define RAISE_SYNTAX_ERROR_INVALID_TARGET(type, e) NULL
#define RAISE_SYNTAX_ERROR_STARTING_FROM(a, msg, ...) NULL
#define RAISE_SYNTAX_ERROR_KNOWN_LOCATION_FROM_TOKEN(a, msg, ...) NULL
#define RAISE_ERROR_KNOWN_LOCATION(p, errtype, lineno, col_offset, end_lineno, \
                                   end_col_offset, msg, ...)                   \
  NULL

#define _PyAST_AnnAssign(target, annotation, value, simple, ...)               \
  ((stmt_ty)lython_cpython_ast_children("AnnAssign", 3, (void *)(target),      \
                                        (void *)(annotation),                  \
                                        (void *)(value)))
#define _PyAST_Assert(test, msg, ...)                                          \
  ((stmt_ty)lython_cpython_ast_children("Assert", 2, (void *)(test),           \
                                        (void *)(msg)))
#define _PyAST_Assign(targets, value, type_comment, ...)                       \
  ((stmt_ty)lython_cpython_ast_children("Assign", 2, (void *)(targets),        \
                                        (void *)(value)))
#define _PyAST_AsyncFor(target, iter, body, orelse, type_comment, ...)         \
  ((stmt_ty)lython_cpython_ast_children("AsyncFor", 4, (void *)(target),       \
                                        (void *)(iter), (void *)(body),        \
                                        (void *)(orelse)))
#define _PyAST_AsyncFunctionDef(name, args, body, decorators, returns,         \
                                type_comment, type_params, ...)                \
  ((stmt_ty)lython_cpython_ast_children(                                       \
      "AsyncFunctionDef", 5, (void *)(args), (void *)(body),                   \
      (void *)(decorators), (void *)(returns), (void *)(type_params)))
#define _PyAST_AsyncWith(items, body, type_comment, ...)                       \
  ((stmt_ty)lython_cpython_ast_children("AsyncWith", 2, (void *)(items),       \
                                        (void *)(body)))
#define _PyAST_Attribute(value, attr, ctx, ...)                                \
  ((expr_ty)lython_cpython_ast_unary("Attribute", (void *)(value)))
#define _PyAST_AugAssign(target, op, value, ...)                               \
  ((stmt_ty)lython_cpython_ast_children("AugAssign", 2, (void *)(target),      \
                                        (void *)(value)))
#define _PyAST_Await(value, ...)                                               \
  ((expr_ty)lython_cpython_ast_unary("Await", (void *)(value)))
#define _PyAST_BinOp(left, op, right, ...)                                     \
  ((expr_ty)lython_cpython_ast_binary("BinOp", (void *)(left), (void *)(right)))
#define _PyAST_BoolOp(op, values, ...)                                         \
  ((expr_ty)lython_cpython_ast_sequence("BoolOp", (asdl_seq *)(values)))
#define _PyAST_Break(...) ((stmt_ty)lython_cpython_ast_stub("Break"))
#define _PyAST_Call(func, args, keywords, ...)                                 \
  lython_cpython_ast_call((expr_ty)(func), (asdl_expr_seq *)(args),            \
                          (asdl_keyword_seq *)(keywords))
#define _PyAST_ClassDef(name, bases, keywords, body, decorators, type_params,  \
                        ...)                                                   \
  ((stmt_ty)lython_cpython_ast_children(                                       \
      "ClassDef", 5, (void *)(bases), (void *)(keywords), (void *)(body),      \
      (void *)(decorators), (void *)(type_params)))
#define _PyAST_Compare(left, ops, comparators, ...)                            \
  ((expr_ty)lython_cpython_ast_children("Compare", 2, (void *)(left),          \
                                        (void *)(comparators)))
#define _PyAST_Constant(...) ((expr_ty)lython_cpython_ast_stub("Constant"))
#define _PyAST_Continue(...) ((stmt_ty)lython_cpython_ast_stub("Continue"))
#define _PyAST_Delete(targets, ...)                                            \
  ((stmt_ty)lython_cpython_ast_sequence("Delete", (asdl_seq *)(targets)))
#define _PyAST_Dict(keys, values, ...)                                         \
  ((expr_ty)lython_cpython_ast_children("Dict", 2, (void *)(keys),             \
                                        (void *)(values)))
#define _PyAST_DictComp(key, value, generators, ...)                           \
  ((expr_ty)lython_cpython_ast_children(                                       \
      "DictComp", 3, (void *)(key), (void *)(value), (void *)(generators)))
#define _PyAST_ExceptHandler(type, name, body, ...)                            \
  ((excepthandler_ty)lython_cpython_ast_children(                              \
      "ExceptHandler", 2, (void *)(type), (void *)(body)))
#define _PyAST_Expr(value, ...)                                                \
  ((stmt_ty)lython_cpython_ast_unary("Expr", (void *)(value)))
#define _PyAST_Expression(body, ...)                                           \
  ((mod_ty)lython_cpython_ast_unary("Expression", (void *)(body)))
#define _PyAST_For(target, iter, body, orelse, type_comment, ...)              \
  ((stmt_ty)lython_cpython_ast_children("For", 4, (void *)(target),            \
                                        (void *)(iter), (void *)(body),        \
                                        (void *)(orelse)))
#define _PyAST_FunctionDef(name, args, body, decorators, returns,              \
                           type_comment, type_params, ...)                     \
  ((stmt_ty)lython_cpython_ast_children(                                       \
      "FunctionDef", 5, (void *)(args), (void *)(body), (void *)(decorators),  \
      (void *)(returns), (void *)(type_params)))
#define _PyAST_FunctionType(argtypes, returns, ...)                            \
  ((mod_ty)lython_cpython_ast_binary("FunctionType", (void *)(argtypes),       \
                                     (void *)(returns)))
#define _PyAST_GeneratorExp(elt, generators, ...)                              \
  ((expr_ty)lython_cpython_ast_children("GeneratorExp", 2, (void *)(elt),      \
                                        (void *)(generators)))
#define _PyAST_Global(...) ((stmt_ty)lython_cpython_ast_stub("Global"))
#define _PyAST_If(test, body, orelse, ...)                                     \
  ((stmt_ty)lython_cpython_ast_children("If", 3, (void *)(test),               \
                                        (void *)(body), (void *)(orelse)))
#define _PyAST_IfExp(test, body, orelse, ...)                                  \
  ((expr_ty)lython_cpython_ast_children("IfExp", 3, (void *)(test),            \
                                        (void *)(body), (void *)(orelse)))
#define _PyAST_Import(names, ...)                                              \
  ((stmt_ty)lython_cpython_ast_sequence("Import", (asdl_seq *)(names)))
#define _PyAST_ImportFrom(module, names, level, ...)                           \
  ((stmt_ty)lython_cpython_ast_sequence("ImportFrom", (asdl_seq *)(names)))
#define _PyAST_Interactive(body, ...)                                          \
  ((mod_ty)lython_cpython_ast_sequence("Interactive", (asdl_seq *)(body)))
#define _PyAST_Lambda(args, body, ...)                                         \
  ((expr_ty)lython_cpython_ast_binary("Lambda", (void *)(args), (void *)(body)))
#define _PyAST_List(elts, ctx, ...)                                            \
  ((expr_ty)lython_cpython_ast_sequence("List", (asdl_seq *)(elts)))
#define _PyAST_ListComp(elt, generators, ...)                                  \
  ((expr_ty)lython_cpython_ast_children("ListComp", 2, (void *)(elt),          \
                                        (void *)(generators)))
#define _PyAST_Match(subject, cases, ...)                                      \
  ((stmt_ty)lython_cpython_ast_children("Match", 2, (void *)(subject),         \
                                        (void *)(cases)))
#define _PyAST_MatchAs(pattern, name, ...)                                     \
  ((pattern_ty)lython_cpython_ast_unary("MatchAs", (void *)(pattern)))
#define _PyAST_MatchClass(cls, patterns, kwd_attrs, kwd_patterns, ...)         \
  ((pattern_ty)lython_cpython_ast_children("MatchClass", 3, (void *)(cls),     \
                                           (void *)(patterns),                 \
                                           (void *)(kwd_patterns)))
#define _PyAST_MatchMapping(keys, patterns, rest, ...)                         \
  ((pattern_ty)lython_cpython_ast_children("MatchMapping", 2, (void *)(keys),  \
                                           (void *)(patterns)))
#define _PyAST_MatchOr(patterns, ...)                                          \
  ((pattern_ty)lython_cpython_ast_sequence("MatchOr", (asdl_seq *)(patterns)))
#define _PyAST_MatchSequence(patterns, ...)                                    \
  ((pattern_ty)lython_cpython_ast_sequence("MatchSequence",                    \
                                           (asdl_seq *)(patterns)))
#define _PyAST_MatchSingleton(...)                                             \
  ((pattern_ty)lython_cpython_ast_stub("MatchSingleton"))
#define _PyAST_MatchStar(...) ((pattern_ty)lython_cpython_ast_stub("MatchStar"))
#define _PyAST_MatchValue(value, ...)                                          \
  ((pattern_ty)lython_cpython_ast_unary("MatchValue", (void *)(value)))
#define _PyAST_NamedExpr(target, value, ...)                                   \
  ((expr_ty)lython_cpython_ast_binary("NamedExpr", (void *)(target),           \
                                      (void *)(value)))
#define _PyAST_Nonlocal(...) ((stmt_ty)lython_cpython_ast_stub("Nonlocal"))
#define _PyAST_ParamSpec(name, default_value, ...)                             \
  ((type_param_ty)lython_cpython_ast_unary("ParamSpec",                        \
                                           (void *)(default_value)))
#define _PyAST_Pass(...) ((stmt_ty)lython_cpython_ast_stub("Pass"))
#define _PyAST_Raise(exc, cause, ...)                                          \
  ((stmt_ty)lython_cpython_ast_children("Raise", 2, (void *)(exc),             \
                                        (void *)(cause)))
#define _PyAST_Return(value, ...)                                              \
  ((stmt_ty)lython_cpython_ast_unary("Return", (void *)(value)))
#define _PyAST_Set(elts, ...)                                                  \
  ((expr_ty)lython_cpython_ast_sequence("Set", (asdl_seq *)(elts)))
#define _PyAST_SetComp(elt, generators, ...)                                   \
  ((expr_ty)lython_cpython_ast_children("SetComp", 2, (void *)(elt),           \
                                        (void *)(generators)))
#define _PyAST_Slice(lower, upper, step, ...)                                  \
  ((expr_ty)lython_cpython_ast_children("Slice", 3, (void *)(lower),           \
                                        (void *)(upper), (void *)(step)))
#define _PyAST_Starred(value, ctx, ...)                                        \
  ((expr_ty)lython_cpython_ast_unary("Starred", (void *)(value)))
#define _PyAST_Subscript(value, slice, ctx, ...)                               \
  ((expr_ty)lython_cpython_ast_binary("Subscript", (void *)(value),            \
                                      (void *)(slice)))
#define _PyAST_Try(body, handlers, orelse, finalbody, ...)                     \
  ((stmt_ty)lython_cpython_ast_children("Try", 4, (void *)(body),              \
                                        (void *)(handlers), (void *)(orelse),  \
                                        (void *)(finalbody)))
#define _PyAST_TryStar(body, handlers, orelse, finalbody, ...)                 \
  ((stmt_ty)lython_cpython_ast_children("TryStar", 4, (void *)(body),          \
                                        (void *)(handlers), (void *)(orelse),  \
                                        (void *)(finalbody)))
#define _PyAST_Tuple(elts, ctx, ...)                                           \
  ((expr_ty)lython_cpython_ast_sequence("Tuple", (asdl_seq *)(elts)))
#define _PyAST_TypeAlias(name, type_params, value, ...)                        \
  ((stmt_ty)lython_cpython_ast_children(                                       \
      "TypeAlias", 3, (void *)(name), (void *)(type_params), (void *)(value)))
#define _PyAST_TypeVar(name, bound, default_value, ...)                        \
  ((type_param_ty)lython_cpython_ast_children("TypeVar", 2, (void *)(bound),   \
                                              (void *)(default_value)))
#define _PyAST_TypeVarTuple(name, default_value, ...)                          \
  ((type_param_ty)lython_cpython_ast_unary("TypeVarTuple",                     \
                                           (void *)(default_value)))
#define _PyAST_UnaryOp(op, operand, ...)                                       \
  ((expr_ty)lython_cpython_ast_unary("UnaryOp", (void *)(operand)))
#define _PyAST_While(test, body, orelse, ...)                                  \
  ((stmt_ty)lython_cpython_ast_children("While", 3, (void *)(test),            \
                                        (void *)(body), (void *)(orelse)))
#define _PyAST_With(items, body, type_comment, ...)                            \
  ((stmt_ty)lython_cpython_ast_children("With", 2, (void *)(items),            \
                                        (void *)(body)))
#define _PyAST_Yield(value, ...)                                               \
  ((expr_ty)lython_cpython_ast_unary("Yield", (void *)(value)))
#define _PyAST_YieldFrom(value, ...)                                           \
  ((expr_ty)lython_cpython_ast_unary("YieldFrom", (void *)(value)))
#define _PyAST_alias(...) ((alias_ty)lython_cpython_ast_stub("alias"))
#define _PyAST_arg(arg, annotation, type_comment, ...)                         \
  ((arg_ty)lython_cpython_ast_unary("arg", (void *)(annotation)))
#define _PyAST_comprehension(target, iter, ifs, is_async, ...)                 \
  ((comprehension_ty)lython_cpython_ast_children(                              \
      "comprehension", 3, (void *)(target), (void *)(iter), (void *)(ifs)))
#define _PyAST_keyword(arg, value, ...)                                        \
  ((keyword_ty)lython_cpython_ast_unary("keyword", (void *)(value)))
#define _PyAST_match_case(pattern, guard, body, ...)                           \
  ((match_case_ty)lython_cpython_ast_children(                                 \
      "match_case", 3, (void *)(pattern), (void *)(guard), (void *)(body)))
#define _PyAST_withitem(context_expr, optional_vars, ...)                      \
  ((withitem_ty)lython_cpython_ast_children(                                   \
      "withitem", 2, (void *)(context_expr), (void *)(optional_vars)))

void *lython_cpython_ast_stub(const char *kind);
void *lython_cpython_ast_sequence(const char *kind, asdl_seq *seq);
void *lython_cpython_ast_unary(const char *kind, void *primary);
void *lython_cpython_ast_binary(const char *kind, void *primary,
                                void *secondary);
void *lython_cpython_ast_children(const char *kind, int child_count, ...);
expr_ty lython_cpython_ast_call(expr_ty func, asdl_expr_seq *args,
                                asdl_keyword_seq *keywords);
int lython_cpython_generated_parser_is_linked(void);
int lython_cpython_generated_parse_tokens(const LythonCpythonToken *tokens,
                                          size_t count, int start_rule,
                                          size_t type_ignore_count);
int lython_cpython_generated_last_mark(void);
int lython_cpython_generated_last_error_indicator(void);
const char *lython_cpython_generated_last_error_source(void);
const char *lython_cpython_generated_last_root_kind(void);
int lython_cpython_generated_last_root_child_count(void);
const char *lython_cpython_generated_last_root_child_kinds(void);
const char *lython_cpython_generated_last_root_child_shapes(void);
const char *lython_cpython_generated_last_root_recursive_shape(void);

int _PyPegen_insert_memo(Parser *p, int mark, int type, void *node);
int _PyPegen_update_memo(Parser *p, int mark, int type, void *node);
int _PyPegen_is_memoized(Parser *p, int type, void *pres);
int _PyPegen_lookahead(int positive, void *(func)(Parser *), Parser *p);
int _PyPegen_lookahead_for_expr(int positive, expr_ty(func)(Parser *),
                                Parser *p);
int _PyPegen_lookahead_for_stmt(int positive, stmt_ty(func)(Parser *),
                                Parser *p);
int _PyPegen_lookahead_with_int(int positive, Token *(func)(Parser *, int),
                                Parser *p, int arg);
int _PyPegen_lookahead_with_string(int positive,
                                   expr_ty(func)(Parser *, const char *),
                                   Parser *p, const char *arg);
Token *_PyPegen_expect_token(Parser *p, int type);
void *_PyPegen_expect_forced_result(Parser *p, void *result,
                                    const char *expected);
Token *_PyPegen_expect_forced_token(Parser *p, int type, const char *expected);
expr_ty _PyPegen_expect_soft_keyword(Parser *p, const char *keyword);
expr_ty _PyPegen_soft_keyword_token(Parser *p);
expr_ty _PyPegen_fstring_middle_token(Parser *p);
asdl_stmt_seq *_PyPegen_interactive_exit(Parser *p);
Token *_PyPegen_get_last_nonnwhitespace_token(Parser *p);
int _PyPegen_fill_token(Parser *p);
expr_ty _PyPegen_name_token(Parser *p);
expr_ty _PyPegen_number_token(Parser *p);
void *_PyPegen_string_token(Parser *p);
PyObject *_PyPegen_set_source_in_metadata(Parser *p, Token *t);
void *_PyPegen_raise_error(Parser *p, PyObject *errtype, int use_mark,
                           const char *errmsg, ...);
void *_PyPegen_raise_error_known_location(Parser *p, PyObject *errtype,
                                          int lineno, int col_offset,
                                          int end_lineno, int end_col_offset,
                                          const char *errmsg, ...);
void _Pypegen_set_syntax_error(Parser *p, Token *last_token);
void _Pypegen_stack_overflow(Parser *p);
void *_PyPegen_dummy_name(Parser *p, ...);
void *_PyPegen_seq_last_item(asdl_seq *seq);
void *_PyPegen_seq_first_item(asdl_seq *seq);
#define PyPegen_last_item(seq, type)                                           \
  ((type)_PyPegen_seq_last_item((asdl_seq *)seq))
#define PyPegen_first_item(seq, type)                                          \
  ((type)_PyPegen_seq_first_item((asdl_seq *)seq))
PyObject *_PyPegen_new_type_comment(Parser *p, const char *s);
arg_ty _PyPegen_add_type_comment_to_arg(Parser *p, arg_ty arg, Token *tc);
PyObject *_PyPegen_new_identifier(Parser *p, const char *n);
asdl_seq *_PyPegen_singleton_seq(Parser *p, void *a);
asdl_seq *_PyPegen_seq_insert_in_front(Parser *p, void *a, asdl_seq *seq);
asdl_seq *_PyPegen_seq_append_to_end(Parser *p, asdl_seq *seq, void *a);
asdl_seq *_PyPegen_seq_flatten(Parser *p, asdl_seq *seq);
expr_ty _PyPegen_join_names_with_dot(Parser *p, expr_ty first, expr_ty second);
int _PyPegen_seq_count_dots(asdl_seq *seq);
alias_ty _PyPegen_alias_for_star(Parser *p, int lineno, int col_offset,
                                 int end_lineno, int end_col_offset,
                                 PyArena *arena);
asdl_identifier_seq *_PyPegen_map_names_to_ids(Parser *p, asdl_expr_seq *seq);
CmpopExprPair *_PyPegen_cmpop_expr_pair(Parser *p, cmpop_ty cmpop,
                                        expr_ty expr);
asdl_int_seq *_PyPegen_get_cmpops(Parser *p, asdl_seq *seq);
asdl_expr_seq *_PyPegen_get_exprs(Parser *p, asdl_seq *seq);
expr_ty _PyPegen_set_expr_context(Parser *p, expr_ty expr, expr_context_ty ctx);
KeyValuePair *_PyPegen_key_value_pair(Parser *p, expr_ty key, expr_ty value);
asdl_expr_seq *_PyPegen_get_keys(Parser *p, asdl_seq *seq);
asdl_expr_seq *_PyPegen_get_values(Parser *p, asdl_seq *seq);
KeyPatternPair *_PyPegen_key_pattern_pair(Parser *p, expr_ty key,
                                          pattern_ty pattern);
asdl_expr_seq *_PyPegen_get_pattern_keys(Parser *p, asdl_seq *seq);
asdl_pattern_seq *_PyPegen_get_patterns(Parser *p, asdl_seq *seq);
NameDefaultPair *_PyPegen_name_default_pair(Parser *p, arg_ty arg,
                                            expr_ty value, Token *tc);
SlashWithDefault *_PyPegen_slash_with_default(Parser *p,
                                              asdl_arg_seq *plain_names,
                                              asdl_seq *names_with_defaults);
StarEtc *_PyPegen_star_etc(Parser *p, arg_ty vararg, asdl_seq *kwonlyargs,
                           arg_ty kwarg);
arguments_ty _PyPegen_make_arguments(Parser *p, asdl_arg_seq *slash_without,
                                     SlashWithDefault *slash_with,
                                     asdl_arg_seq *plain_names,
                                     asdl_seq *names_with_defaults,
                                     StarEtc *star_etc);
arguments_ty _PyPegen_empty_arguments(Parser *p);
expr_ty _PyPegen_template_str(Parser *p, Token *a,
                              asdl_expr_seq *raw_expressions, Token *b);
expr_ty _PyPegen_joined_str(Parser *p, Token *a, asdl_expr_seq *raw_expressions,
                            Token *b);
expr_ty _PyPegen_interpolation(Parser *p, expr_ty a, Token *b,
                               ResultTokenWithMetadata *c,
                               ResultTokenWithMetadata *d, Token *e, int l,
                               int c0, int el, int ec, PyArena *arena);
expr_ty _PyPegen_formatted_value(Parser *p, expr_ty a, Token *b,
                                 ResultTokenWithMetadata *c,
                                 ResultTokenWithMetadata *d, Token *e, int l,
                                 int c0, int el, int ec, PyArena *arena);
void *_PyPegen_augoperator(Parser *p, operator_ty type);
stmt_ty _PyPegen_function_def_decorators(Parser *p, asdl_expr_seq *decorators,
                                         stmt_ty function_def);
stmt_ty _PyPegen_class_def_decorators(Parser *p, asdl_expr_seq *decorators,
                                      stmt_ty class_def);
KeywordOrStarred *_PyPegen_keyword_or_starred(Parser *p, void *element,
                                              int is_keyword);
asdl_expr_seq *_PyPegen_seq_extract_starred_exprs(Parser *p, asdl_seq *seq);
asdl_keyword_seq *_PyPegen_seq_delete_starred_exprs(Parser *p, asdl_seq *seq);
expr_ty _PyPegen_collect_call_seqs(Parser *p, asdl_expr_seq *a, asdl_seq *b,
                                   int lineno, int col_offset, int end_lineno,
                                   int end_col_offset, PyArena *arena);
expr_ty _PyPegen_constant_from_token(Parser *p, Token *tok);
expr_ty _PyPegen_decoded_constant_from_token(Parser *p, Token *tok);
expr_ty _PyPegen_constant_from_string(Parser *p, Token *tok);
expr_ty _PyPegen_concatenate_tstrings(Parser *p, asdl_expr_seq *seq, int l,
                                      int c, int el, int ec, PyArena *arena);
expr_ty _PyPegen_concatenate_strings(Parser *p, asdl_expr_seq *seq, int l,
                                     int c, int el, int ec, PyArena *arena);
expr_ty _PyPegen_FetchRawForm(Parser *p, int l, int c, int el, int ec);
expr_ty _PyPegen_ensure_imaginary(Parser *p, expr_ty e);
expr_ty _PyPegen_ensure_real(Parser *p, expr_ty e);
asdl_seq *_PyPegen_join_sequences(Parser *p, asdl_seq *a, asdl_seq *b);
int _PyPegen_check_barry_as_flufl(Parser *p, Token *t);
int _PyPegen_check_legacy_stmt(Parser *p, expr_ty t);
ResultTokenWithMetadata *_PyPegen_check_fstring_conversion(Parser *p, Token *t,
                                                           expr_ty e);
ResultTokenWithMetadata *
_PyPegen_setup_full_format_spec(Parser *p, Token *t, asdl_expr_seq *seq, int l,
                                int c, int el, int ec, PyArena *arena);
mod_ty _PyPegen_make_module(Parser *p, asdl_stmt_seq *stmts);
void *_PyPegen_arguments_parsing_error(Parser *p, expr_ty e);
expr_ty _PyPegen_get_last_comprehension_item(comprehension_ty comprehension);
void *_PyPegen_nonparen_genexp_in_call(Parser *p, expr_ty args,
                                       asdl_comprehension_seq *comprehensions);
stmt_ty _PyPegen_checked_future_import(Parser *p, identifier module,
                                       asdl_alias_seq *names, int lineno,
                                       int col_offset, int end_lineno,
                                       int end_col_offset, int feature_version,
                                       PyArena *arena);
asdl_stmt_seq *_PyPegen_register_stmts(Parser *p, asdl_stmt_seq *stmts);
stmt_ty _PyPegen_register_stmt(Parser *p, stmt_ty s);
expr_ty _PyPegen_get_invalid_target(expr_ty e, TARGETS_TYPE targets_type);
const char *_PyPegen_get_expr_name(expr_ty e);
void *_PyPegen_parse(Parser *p);

#ifdef __cplusplus
}
#endif

#endif
