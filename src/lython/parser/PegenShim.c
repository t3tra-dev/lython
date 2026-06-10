// Patched CPython 3.14 pegen helper shim for Lython.
//
// These definitions make the generated parser.c link without CPython runtime
// internals. The C++ frontend keeps the public Python-ast-shaped Node API; this
// C layer is the ABI seam where CPython parser actions are incrementally
// replaced by Lython Node builders.

#include "pegen.h"

#include <string.h>

extern void *_PyPegen_parse(Parser *p);

void *lython_cpython_generated_parser_anchor = (void *)&_PyPegen_parse;
static int lython_last_mark = 0;
static int lython_last_error_indicator = 0;
static const char *lython_last_error_source = "none";
static const char *lython_last_root_kind = "none";
static int lython_last_root_child_count = -1;
static char *lython_last_root_child_kinds = NULL;
static char *lython_last_root_child_shapes = NULL;
static char *lython_last_root_recursive_shape = NULL;
static _Thread_local PyArena *lython_current_arena = NULL;
static _Thread_local Token *lython_current_tokens = NULL;
static _Thread_local size_t lython_current_token_count = 0;
static _Thread_local void **lython_current_ast_nodes = NULL;
static _Thread_local size_t lython_current_ast_node_count = 0;
static _Thread_local size_t lython_current_ast_node_capacity = 0;
static _Thread_local asdl_seq **lython_current_asdl_seqs = NULL;
static _Thread_local size_t lython_current_asdl_seq_count = 0;
static _Thread_local size_t lython_current_asdl_seq_capacity = 0;

static void *lython_pegen_alloc(size_t size) {
  void *ptr = calloc(1, size);
  if (!ptr || !lython_current_arena)
    return ptr;
  if (lython_current_arena->size == lython_current_arena->capacity) {
    size_t next_capacity = lython_current_arena->capacity == 0
                               ? 64
                               : lython_current_arena->capacity * 2;
    void **next = (void **)realloc(lython_current_arena->allocations,
                                   next_capacity * sizeof(void *));
    if (!next)
      return ptr;
    lython_current_arena->allocations = next;
    lython_current_arena->capacity = next_capacity;
  }
  lython_current_arena->allocations[lython_current_arena->size++] = ptr;
  return ptr;
}

static void lython_pegen_arena_free(PyArena *arena) {
  if (!arena)
    return;
  for (size_t i = 0; i < arena->size; ++i)
    free(arena->allocations[i]);
  free(arena->allocations);
  arena->allocations = NULL;
  arena->size = 0;
  arena->capacity = 0;
}

int lython_cpython_generated_parser_is_linked(void) {
  return lython_cpython_generated_parser_anchor != NULL;
}

int lython_cpython_generated_last_mark(void) { return lython_last_mark; }

int lython_cpython_generated_last_error_indicator(void) {
  return lython_last_error_indicator;
}

const char *lython_cpython_generated_last_error_source(void) {
  return lython_last_error_source;
}

const char *lython_cpython_generated_last_root_kind(void) {
  return lython_last_root_kind;
}

int lython_cpython_generated_last_root_child_count(void) {
  return lython_last_root_child_count;
}

const char *lython_cpython_generated_last_root_child_kinds(void) {
  return lython_last_root_child_kinds ? lython_last_root_child_kinds : "";
}

const char *lython_cpython_generated_last_root_child_shapes(void) {
  return lython_last_root_child_shapes ? lython_last_root_child_shapes : "";
}

const char *lython_cpython_generated_last_root_recursive_shape(void) {
  return lython_last_root_recursive_shape ? lython_last_root_recursive_shape
                                          : "";
}

static void lython_ast_registry_reset(void) {
  free(lython_current_ast_nodes);
  lython_current_ast_nodes = NULL;
  lython_current_ast_node_count = 0;
  lython_current_ast_node_capacity = 0;
  free(lython_current_asdl_seqs);
  lython_current_asdl_seqs = NULL;
  lython_current_asdl_seq_count = 0;
  lython_current_asdl_seq_capacity = 0;
}

static void lython_register_ast_node(void *node) {
  if (!node)
    return;
  if (lython_current_ast_node_count == lython_current_ast_node_capacity) {
    size_t next_capacity = lython_current_ast_node_capacity == 0
                               ? 64
                               : lython_current_ast_node_capacity * 2;
    void **next = (void **)realloc(lython_current_ast_nodes,
                                   next_capacity * sizeof(void *));
    if (!next)
      return;
    lython_current_ast_nodes = next;
    lython_current_ast_node_capacity = next_capacity;
  }
  lython_current_ast_nodes[lython_current_ast_node_count++] = node;
}

static void lython_register_asdl_seq(asdl_seq *seq) {
  if (!seq)
    return;
  if (lython_current_asdl_seq_count == lython_current_asdl_seq_capacity) {
    size_t next_capacity = lython_current_asdl_seq_capacity == 0
                               ? 64
                               : lython_current_asdl_seq_capacity * 2;
    asdl_seq **next = (asdl_seq **)realloc(lython_current_asdl_seqs,
                                           next_capacity * sizeof(asdl_seq *));
    if (!next)
      return;
    lython_current_asdl_seqs = next;
    lython_current_asdl_seq_capacity = next_capacity;
  }
  lython_current_asdl_seqs[lython_current_asdl_seq_count++] = seq;
}

static int lython_is_registered_asdl_seq(void *value) {
  if (!value)
    return 0;
  for (size_t i = 0; i < lython_current_asdl_seq_count; ++i)
    if ((void *)lython_current_asdl_seqs[i] == value)
      return 1;
  return 0;
}

static const char *lython_ast_kind(void *node) {
  if (!node)
    return "none";
  int registered = 0;
  for (size_t i = 0; i < lython_current_ast_node_count; ++i) {
    if (lython_current_ast_nodes[i] == node) {
      registered = 1;
      break;
    }
  }
  if (!registered)
    return "unknown";
  const struct _expr *ast = (const struct _expr *)node;
  return ast->lython_kind ? ast->lython_kind : "unknown";
}

static int lython_ast_child_count(void *node) {
  if (!node)
    return 0;
  int registered = 0;
  for (size_t i = 0; i < lython_current_ast_node_count; ++i) {
    if (lython_current_ast_nodes[i] == node) {
      registered = 1;
      break;
    }
  }
  if (!registered)
    return -1;
  const struct _expr *ast = (const struct _expr *)node;
  if (ast->lython_kind && strcmp(ast->lython_kind, "FunctionType") == 0) {
    int count = lython_is_registered_asdl_seq(ast->lython_primary)
                    ? (int)asdl_seq_LEN((asdl_seq *)ast->lython_primary)
                    : 0;
    if (ast->lython_secondary)
      ++count;
    return count;
  }
  if (ast->lython_seq)
    return (int)asdl_seq_LEN(ast->lython_seq);
  if (ast->lython_primary && ast->lython_secondary)
    return 2;
  if (ast->lython_primary)
    return 1;
  return 0;
}

static const char *lython_registered_ast_kind(void *node) {
  if (!node)
    return "None";
  for (size_t i = 0; i < lython_current_ast_node_count; ++i) {
    if (lython_current_ast_nodes[i] == node) {
      const struct _expr *ast = (const struct _expr *)node;
      return ast->lython_kind ? ast->lython_kind : "Unknown";
    }
  }
  if (lython_current_tokens && lython_current_token_count > 0) {
    Token *token = (Token *)node;
    if (token >= lython_current_tokens &&
        token < lython_current_tokens + lython_current_token_count) {
      switch (token->type) {
      case NUMBER:
      case STRING:
      case FSTRING_MIDDLE:
      case TSTRING_MIDDLE:
        return "Constant";
      case NAME:
      case SOFT_KEYWORD:
        return "Name";
      default:
        return "Token";
      }
    }
  }
  return "Unknown";
}

static void lython_digest_append(char **buffer, size_t *size, size_t *capacity,
                                 const char *text) {
  if (!text)
    text = "";
  size_t text_len = strlen(text);
  size_t needed = *size + text_len + 1;
  if (needed > *capacity) {
    size_t next_capacity = *capacity == 0 ? 64 : *capacity;
    while (next_capacity < needed)
      next_capacity *= 2;
    char *next = (char *)realloc(*buffer, next_capacity);
    if (!next)
      return;
    *buffer = next;
    *capacity = next_capacity;
  }
  memcpy(*buffer + *size, text, text_len);
  *size += text_len;
  (*buffer)[*size] = '\0';
}

static void lython_digest_append_kind(char **buffer, size_t *size,
                                      size_t *capacity, int *first,
                                      const char *kind) {
  if (!*first)
    lython_digest_append(buffer, size, capacity, ",");
  lython_digest_append(buffer, size, capacity, kind);
  *first = 0;
}

static asdl_seq *lython_root_sequence_for_digest(void *node) {
  if (!node)
    return NULL;
  for (size_t i = 0; i < lython_current_ast_node_count; ++i) {
    if (lython_current_ast_nodes[i] != node)
      continue;
    const struct _expr *ast = (const struct _expr *)node;
    if (ast->lython_kind && strcmp(ast->lython_kind, "FunctionType") == 0 &&
        lython_is_registered_asdl_seq(ast->lython_primary))
      return (asdl_seq *)ast->lython_primary;
    return ast->lython_seq;
  }
  return NULL;
}

static const struct _expr *pegen_registered_ast(void *node) {
  if (!node)
    return NULL;
  for (size_t i = 0; i < lython_current_ast_node_count; ++i)
    if (lython_current_ast_nodes[i] == node)
      return (const struct _expr *)node;
  return NULL;
}

static void pegen_digest_sequence_child_kinds(char **buffer, size_t *size,
                                              size_t *capacity, int *first,
                                              asdl_seq *seq) {
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i)
    lython_digest_append_kind(buffer, size, capacity, first,
                              lython_registered_ast_kind(seq->elements[i]));
}

static int pegen_digest_func_type_kinds(char **buffer, size_t *size,
                                        size_t *capacity, int *first,
                                        const struct _expr *ast) {
  if (!ast || !ast->lython_kind ||
      strcmp(ast->lython_kind, "FunctionType") != 0)
    return 0;
  if (lython_is_registered_asdl_seq(ast->lython_primary))
    pegen_digest_sequence_child_kinds(buffer, size, capacity, first,
                                      (asdl_seq *)ast->lython_primary);
  if (ast->lython_secondary)
    lython_digest_append_kind(
        buffer, size, capacity, first,
        lython_registered_ast_kind(ast->lython_secondary));
  return 1;
}

static void pegen_digest_pair_kinds(char **buffer, size_t *size,
                                    size_t *capacity, int *first,
                                    const struct _expr *ast) {
  if (!ast)
    return;
  if (ast->lython_primary)
    lython_digest_append_kind(buffer, size, capacity, first,
                              lython_registered_ast_kind(ast->lython_primary));
  if (ast->lython_secondary)
    lython_digest_append_kind(
        buffer, size, capacity, first,
        lython_registered_ast_kind(ast->lython_secondary));
}

static char *lython_ast_child_kind_digest(void *node) {
  char *buffer = NULL;
  size_t size = 0;
  size_t capacity = 0;
  int first = 1;
  const struct _expr *ast = pegen_registered_ast(node);

  if (pegen_digest_func_type_kinds(&buffer, &size, &capacity, &first, ast))
    goto done;

  asdl_seq *seq = lython_root_sequence_for_digest(node);
  if (seq)
    pegen_digest_sequence_child_kinds(&buffer, &size, &capacity, &first, seq);
  else
    pegen_digest_pair_kinds(&buffer, &size, &capacity, &first, ast);

done:
  if (!buffer)
    buffer = (char *)calloc(1, 1);
  return buffer;
}

static const struct _expr *lython_registered_ast_node(void *node) {
  return pegen_registered_ast(node);
}

static void lython_append_immediate_child_kinds(char **buffer, size_t *size,
                                                size_t *capacity, void *node) {
  const struct _expr *ast = lython_registered_ast_node(node);
  if (!ast)
    return;

  int first = 1;
  int ignore_string_constants =
      ast->lython_kind && (strcmp(ast->lython_kind, "JoinedStr") == 0 ||
                           strcmp(ast->lython_kind, "TemplateStr") == 0);
  if (ast->lython_seq) {
    for (Py_ssize_t i = 0; i < asdl_seq_LEN(ast->lython_seq); ++i) {
      const char *kind =
          lython_registered_ast_kind(ast->lython_seq->elements[i]);
      if (ignore_string_constants && strcmp(kind, "Constant") == 0)
        continue;
      lython_digest_append_kind(buffer, size, capacity, &first, kind);
    }
    return;
  }
  if (ast->lython_primary) {
    const char *kind = lython_registered_ast_kind(ast->lython_primary);
    if (!ignore_string_constants || strcmp(kind, "Constant") != 0)
      lython_digest_append_kind(buffer, size, capacity, &first, kind);
  }
  if (ast->lython_secondary) {
    const char *kind = lython_registered_ast_kind(ast->lython_secondary);
    if (!ignore_string_constants || strcmp(kind, "Constant") != 0)
      lython_digest_append_kind(buffer, size, capacity, &first, kind);
  }
}

static void lython_digest_append_shape(char **buffer, size_t *size,
                                       size_t *capacity, int *first,
                                       void *node) {
  if (!*first)
    lython_digest_append(buffer, size, capacity, ";");
  lython_digest_append(buffer, size, capacity,
                       lython_registered_ast_kind(node));

  char *children = NULL;
  size_t child_size = 0;
  size_t child_capacity = 0;
  lython_append_immediate_child_kinds(&children, &child_size, &child_capacity,
                                      node);
  if (children && children[0] != '\0') {
    lython_digest_append(buffer, size, capacity, "(");
    lython_digest_append(buffer, size, capacity, children);
    lython_digest_append(buffer, size, capacity, ")");
  }
  free(children);
  *first = 0;
}

static void pegen_append_func_type_shapes(char **buffer, size_t *size,
                                          size_t *capacity, int *first,
                                          const struct _expr *ast) {
  if (lython_is_registered_asdl_seq(ast->lython_primary)) {
    const asdl_seq *argtypes = (const asdl_seq *)ast->lython_primary;
    for (Py_ssize_t arg = 0; arg < asdl_seq_LEN(argtypes); ++arg)
      lython_digest_append_shape(buffer, size, capacity, first,
                                 argtypes->elements[arg]);
  }
  if (ast->lython_secondary)
    lython_digest_append_shape(buffer, size, capacity, first,
                               ast->lython_secondary);
}

static char *lython_ast_root_child_shape_digest(void *node) {
  char *buffer = NULL;
  size_t size = 0;
  size_t capacity = 0;
  int first = 1;

  const struct _expr *ast = lython_registered_ast_node(node);
  if (!ast) {
    buffer = (char *)calloc(1, 1);
    return buffer;
  }

  if (ast->lython_kind && strcmp(ast->lython_kind, "FunctionType") == 0) {
    pegen_append_func_type_shapes(&buffer, &size, &capacity, &first, ast);
  } else if (ast->lython_seq) {
    for (Py_ssize_t i = 0; i < asdl_seq_LEN(ast->lython_seq); ++i)
      lython_digest_append_shape(&buffer, &size, &capacity, &first,
                                 ast->lython_seq->elements[i]);
  } else {
    if (ast->lython_primary)
      lython_digest_append_shape(&buffer, &size, &capacity, &first,
                                 ast->lython_primary);
    if (ast->lython_secondary)
      lython_digest_append_shape(&buffer, &size, &capacity, &first,
                                 ast->lython_secondary);
  }

  if (!buffer)
    buffer = (char *)calloc(1, 1);
  return buffer;
}

static void lython_append_recursive_shape(char **buffer, size_t *size,
                                          size_t *capacity, void *node);

static void lython_append_recursive_shape_list(char **buffer, size_t *size,
                                               size_t *capacity, int *first,
                                               asdl_seq *seq) {
  if (!seq)
    return;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    if (!seq->elements[i])
      continue;
    if (!*first)
      lython_digest_append(buffer, size, capacity, ",");
    lython_append_recursive_shape(buffer, size, capacity, seq->elements[i]);
    *first = 0;
  }
}

static void lython_append_recursive_shape_child(char **buffer, size_t *size,
                                                size_t *capacity, int *first,
                                                void *node) {
  if (!node)
    return;
  if (!*first)
    lython_digest_append(buffer, size, capacity, ",");
  lython_append_recursive_shape(buffer, size, capacity, node);
  *first = 0;
}

static void pegen_append_func_type_recursive(char **children,
                                             size_t *child_size,
                                             size_t *child_capacity, int *first,
                                             const struct _expr *ast) {
  if (lython_is_registered_asdl_seq(ast->lython_primary))
    lython_append_recursive_shape_list(children, child_size, child_capacity,
                                       first, (asdl_seq *)ast->lython_primary);
  lython_append_recursive_shape_child(children, child_size, child_capacity,
                                      first, ast->lython_secondary);
}

static void pegen_append_recursive_ast_children(char **children,
                                                size_t *child_size,
                                                size_t *child_capacity,
                                                int *first,
                                                const struct _expr *ast) {
  if (ast->lython_kind && strcmp(ast->lython_kind, "FunctionType") == 0) {
    pegen_append_func_type_recursive(children, child_size, child_capacity,
                                     first, ast);
    return;
  }
  if (ast->lython_seq) {
    lython_append_recursive_shape_list(children, child_size, child_capacity,
                                       first, ast->lython_seq);
    return;
  }
  lython_append_recursive_shape_child(children, child_size, child_capacity,
                                      first, ast->lython_primary);
  lython_append_recursive_shape_child(children, child_size, child_capacity,
                                      first, ast->lython_secondary);
}

static void lython_append_recursive_shape(char **buffer, size_t *size,
                                          size_t *capacity, void *node) {
  const struct _expr *ast = lython_registered_ast_node(node);
  if (!ast) {
    lython_digest_append(buffer, size, capacity,
                         lython_registered_ast_kind(node));
    return;
  }

  lython_digest_append(buffer, size, capacity,
                       ast->lython_kind ? ast->lython_kind : "Unknown");
  if (ast->lython_kind && strcmp(ast->lython_kind, "Constant") == 0)
    return;

  char *children = NULL;
  size_t child_size = 0;
  size_t child_capacity = 0;
  int first = 1;
  pegen_append_recursive_ast_children(&children, &child_size, &child_capacity,
                                      &first, ast);

  if (children && children[0] != '\0') {
    lython_digest_append(buffer, size, capacity, "(");
    lython_digest_append(buffer, size, capacity, children);
    lython_digest_append(buffer, size, capacity, ")");
  }
  free(children);
}

static char *lython_ast_recursive_shape_digest(void *node) {
  char *buffer = NULL;
  size_t size = 0;
  size_t capacity = 0;
  lython_append_recursive_shape(&buffer, &size, &capacity, node);
  if (!buffer)
    buffer = (char *)calloc(1, 1);
  return buffer;
}

typedef struct {
  Token *records;
  Token **tokens;
  size_t capacity;
} PegenTokenBuffer;

static int pegen_token_buffer_init(PegenTokenBuffer *buffer,
                                   const LythonCpythonToken *input,
                                   size_t count) {
  if (!buffer || !input || count == 0)
    return 0;
  buffer->capacity = count + 32;
  buffer->records = (Token *)calloc(buffer->capacity, sizeof(Token));
  buffer->tokens = (Token **)calloc(buffer->capacity, sizeof(Token *));
  if (!buffer->records || !buffer->tokens)
    return 0;

  for (size_t i = 0; i < buffer->capacity; ++i) {
    const LythonCpythonToken *source =
        i < count ? &input[i] : &input[count - 1];
    buffer->records[i].type = source->type;
    buffer->records[i].bytes = (PyObject *)source->text;
    buffer->records[i].metadata = (PyObject *)source->text;
    buffer->records[i].lineno = source->lineno;
    buffer->records[i].col_offset = source->col_offset;
    buffer->records[i].end_lineno = source->end_lineno;
    buffer->records[i].end_col_offset = source->end_col_offset;
    buffer->tokens[i] = &buffer->records[i];
  }
  return 1;
}

static void pegen_token_buffer_destroy(PegenTokenBuffer *buffer) {
  if (!buffer)
    return;
  free(buffer->tokens);
  free(buffer->records);
  buffer->tokens = NULL;
  buffer->records = NULL;
  buffer->capacity = 0;
}

static void pegen_reset_last_result(void) {
  lython_last_error_source = "none";
  lython_last_root_kind = "none";
  lython_last_root_child_count = -1;
  free(lython_last_root_child_kinds);
  lython_last_root_child_kinds = NULL;
  free(lython_last_root_child_shapes);
  lython_last_root_child_shapes = NULL;
  free(lython_last_root_recursive_shape);
  lython_last_root_recursive_shape = NULL;
}

static void pegen_parser_init(Parser *parser, PyArena *arena,
                              const PegenTokenBuffer *buffer, size_t count,
                              int start_rule, size_t type_ignore_count) {
  memset(parser, 0, sizeof(*parser));
  parser->tokens = buffer->tokens;
  parser->fill = (int)count;
  parser->size = (int)buffer->capacity;
  parser->arena = arena;
  parser->start_rule = start_rule;
  parser->feature_version = 14;
  parser->type_ignore_comments.size = type_ignore_count;
  parser->type_ignore_comments.num_items = type_ignore_count;
}

static void pegen_enter_session(PyArena *arena,
                                const PegenTokenBuffer *buffer) {
  lython_current_arena = arena;
  lython_current_tokens = buffer->records;
  lython_current_token_count = buffer->capacity;
  lython_ast_registry_reset();
}

static void pegen_leave_session(PyArena *arena) {
  lython_ast_registry_reset();
  lython_current_arena = NULL;
  lython_current_tokens = NULL;
  lython_current_token_count = 0;
  lython_pegen_arena_free(arena);
}

static void pegen_publish_result(Parser *parser, void *result) {
  lython_last_mark = parser->mark;
  lython_last_error_indicator = parser->error_indicator;
  lython_last_root_kind = lython_ast_kind(result);
  lython_last_root_child_count = lython_ast_child_count(result);
  lython_last_root_child_kinds = lython_ast_child_kind_digest(result);
  lython_last_root_child_shapes = lython_ast_root_child_shape_digest(result);
  lython_last_root_recursive_shape = lython_ast_recursive_shape_digest(result);
}

int lython_cpython_generated_parse_tokens(const LythonCpythonToken *input,
                                          size_t count, int start_rule,
                                          size_t type_ignore_count) {
  if (!input || count == 0)
    return 0;

  PegenTokenBuffer buffer = {0};
  if (!pegen_token_buffer_init(&buffer, input, count)) {
    pegen_token_buffer_destroy(&buffer);
    return 0;
  }

  PyArena arena = {0};
  Parser parser;
  pegen_reset_last_result();
  pegen_parser_init(&parser, &arena, &buffer, count, start_rule,
                    type_ignore_count);
  pegen_enter_session(&arena, &buffer);
  void *result = _PyPegen_parse(&parser);
  pegen_publish_result(&parser, result);
  int success = result != NULL && parser.error_indicator == 0;

  pegen_leave_session(&arena);
  pegen_token_buffer_destroy(&buffer);
  return success;
}

asdl_seq *asdl_generic_seq_new(Py_ssize_t size, PyArena *arena) {
  UNUSED(arena);
  if (size < 0)
    return NULL;
  asdl_seq *seq = (asdl_seq *)lython_pegen_alloc(sizeof(asdl_seq));
  if (!seq)
    return NULL;
  seq->size = size;
  if (size == 0) {
    lython_register_asdl_seq(seq);
    return seq;
  }
  seq->elements = (void **)lython_pegen_alloc((size_t)size * sizeof(void *));
  if (!seq->elements) {
    return NULL;
  }
  lython_register_asdl_seq(seq);
  return seq;
}

void *lython_cpython_ast_stub(const char *kind) {
  struct _expr *node = (struct _expr *)lython_pegen_alloc(sizeof(struct _expr));
  if (!node)
    return NULL;
  if (kind && strcmp(kind, "Name") == 0)
    node->kind = Name_kind;
  else if (kind && strcmp(kind, "Call") == 0)
    node->kind = Call_kind;
  else if (kind && strcmp(kind, "Tuple") == 0)
    node->kind = Tuple_kind;
  node->lython_kind = kind ? kind : "Unknown";
  lython_register_ast_node(node);
  return node;
}

void *lython_cpython_ast_sequence(const char *kind, asdl_seq *seq) {
  struct _expr *node = (struct _expr *)lython_cpython_ast_stub(kind);
  if (node)
    node->lython_seq = seq;
  return node;
}

void *lython_cpython_ast_unary(const char *kind, void *primary) {
  struct _expr *node = (struct _expr *)lython_cpython_ast_stub(kind);
  if (node)
    node->lython_primary = primary;
  return node;
}

void *lython_cpython_ast_binary(const char *kind, void *primary,
                                void *secondary) {
  struct _expr *node = (struct _expr *)lython_cpython_ast_unary(kind, primary);
  if (node)
    node->lython_secondary = secondary;
  return node;
}

static Py_ssize_t pegen_flattened_child_count(void **children,
                                              int child_count) {
  Py_ssize_t count = 0;
  for (int i = 0; i < child_count; ++i) {
    if (lython_is_registered_asdl_seq(children[i]))
      count += asdl_seq_LEN((asdl_seq *)children[i]);
    else if (children[i])
      ++count;
  }
  return count;
}

static void pegen_append_flattened_children(asdl_seq *target, void **children,
                                            int child_count) {
  Py_ssize_t out = 0;
  for (int i = 0; i < child_count; ++i) {
    if (lython_is_registered_asdl_seq(children[i])) {
      asdl_seq *nested = (asdl_seq *)children[i];
      for (Py_ssize_t j = 0; j < asdl_seq_LEN(nested); ++j)
        target->elements[out++] = nested->elements[j];
      continue;
    }
    if (children[i])
      target->elements[out++] = children[i];
  }
}

void *lython_cpython_ast_children(const char *kind, int child_count, ...) {
  if (child_count < 0)
    return NULL;

  void **children = NULL;
  if (child_count > 0) {
    children =
        (void **)lython_pegen_alloc((size_t)child_count * sizeof(void *));
    if (!children)
      return NULL;
  }

  va_list args;
  va_start(args, child_count);
  for (int i = 0; i < child_count; ++i)
    children[i] = va_arg(args, void *);
  va_end(args);

  asdl_seq *seq = asdl_generic_seq_new(
      pegen_flattened_child_count(children, child_count), NULL);
  if (!seq)
    return NULL;
  pegen_append_flattened_children(seq, children, child_count);
  return lython_cpython_ast_sequence(kind, seq);
}

expr_ty lython_cpython_ast_call(expr_ty func, asdl_expr_seq *args,
                                asdl_keyword_seq *keywords) {
  expr_ty node = (expr_ty)lython_cpython_ast_children(
      "Call", 3, (void *)func, (void *)args, (void *)keywords);
  if (!node)
    return NULL;
  node->v.Call.args = args;
  node->v.Call.keywords = keywords;
  return node;
}

static void lython_ast_append_sequence_children(void *node, asdl_seq *extra) {
  struct _expr *ast = (struct _expr *)lython_registered_ast_node(node);
  if (!ast || !extra || asdl_seq_LEN(extra) == 0)
    return;

  Py_ssize_t old_size = asdl_seq_LEN(ast->lython_seq);
  Py_ssize_t extra_size = asdl_seq_LEN(extra);
  asdl_seq *combined = asdl_generic_seq_new(old_size + extra_size, NULL);
  if (!combined)
    return;

  Py_ssize_t out = 0;
  if (ast->lython_seq) {
    for (Py_ssize_t i = 0; i < asdl_seq_LEN(ast->lython_seq); ++i)
      combined->elements[out++] = ast->lython_seq->elements[i];
  }
  for (Py_ssize_t i = 0; i < extra_size; ++i)
    combined->elements[out++] = extra->elements[i];
  ast->lython_seq = combined;
}

static expr_ty lython_ast_name_from_token(Token *token) {
  if (!token)
    return NULL;
  expr_ty node = (expr_ty)lython_cpython_ast_stub("Name");
  if (!node)
    return NULL;
  node->lineno = token->lineno;
  node->col_offset = token->col_offset;
  node->end_lineno = token->end_lineno;
  node->end_col_offset = token->end_col_offset;
  node->v.Name.id = token->bytes;
  return node;
}

static const char *lython_token_text(Token *token) {
  return token ? (const char *)token->bytes : NULL;
}

static int lython_is_soft_keyword_text(const char *text) {
  return text && (strcmp(text, "match") == 0 || strcmp(text, "case") == 0 ||
                  strcmp(text, "type") == 0 || strcmp(text, "_") == 0);
}

int _PyPegen_insert_memo(Parser *p, int mark, int type, void *node) {
  if (!p || !p->tokens || mark < 0 || mark >= p->size)
    return -1;
  Memo *memo = (Memo *)lython_pegen_alloc(sizeof(Memo));
  if (!memo)
    return -1;
  memo->type = type;
  memo->node = node;
  memo->mark = p->mark;
  memo->next = p->tokens[mark]->memo;
  p->tokens[mark]->memo = memo;
  return 0;
}

int _PyPegen_update_memo(Parser *p, int mark, int type, void *node) {
  if (!p || !p->tokens || mark < 0 || mark >= p->size)
    return -1;
  for (Memo *memo = p->tokens[mark]->memo; memo; memo = memo->next) {
    if (memo->type == type) {
      memo->node = node;
      memo->mark = p->mark;
      return 0;
    }
  }
  return _PyPegen_insert_memo(p, mark, type, node);
}

int _PyPegen_is_memoized(Parser *p, int type, void *pres) {
  if (!p || !p->tokens || p->mark < 0 || p->mark >= p->size)
    return 0;
  for (Memo *memo = p->tokens[p->mark]->memo; memo; memo = memo->next) {
    if (memo->type != type)
      continue;
    if (pres)
      *(void **)pres = memo->node;
    p->mark = memo->mark;
    return 1;
  }
  return 0;
}

int _PyPegen_lookahead(int positive, void *(func)(Parser *), Parser *p) {
  int mark = p ? p->mark : 0;
  void *result = func ? func(p) : NULL;
  if (p)
    p->mark = mark;
  return positive ? result != NULL : result == NULL;
}

int _PyPegen_lookahead_for_expr(int positive, expr_ty(func)(Parser *),
                                Parser *p) {
  return _PyPegen_lookahead(positive, (void *(*)(Parser *))func, p);
}

int _PyPegen_lookahead_for_stmt(int positive, stmt_ty(func)(Parser *),
                                Parser *p) {
  return _PyPegen_lookahead(positive, (void *(*)(Parser *))func, p);
}

int _PyPegen_lookahead_with_int(int positive, Token *(func)(Parser *, int),
                                Parser *p, int arg) {
  int mark = p ? p->mark : 0;
  void *result = func ? func(p, arg) : NULL;
  if (p)
    p->mark = mark;
  return positive ? result != NULL : result == NULL;
}

int _PyPegen_lookahead_with_string(int positive,
                                   expr_ty(func)(Parser *, const char *),
                                   Parser *p, const char *arg) {
  int mark = p ? p->mark : 0;
  void *result = func ? func(p, arg) : NULL;
  if (p)
    p->mark = mark;
  return positive ? result != NULL : result == NULL;
}

Token *_PyPegen_expect_token(Parser *p, int type) {
  if (!p || p->mark >= p->fill || !p->tokens)
    return NULL;
  Token *token = p->tokens[p->mark];
  if (!token || token->type != type)
    return NULL;
  ++p->mark;
  return token;
}

void *_PyPegen_expect_forced_result(Parser *p, void *result,
                                    const char *expected) {
  UNUSED(p);
  UNUSED(expected);
  return result;
}

Token *_PyPegen_expect_forced_token(Parser *p, int type, const char *expected) {
  UNUSED(expected);
  return _PyPegen_expect_token(p, type);
}

expr_ty _PyPegen_expect_soft_keyword(Parser *p, const char *keyword) {
  if (!p || p->mark >= p->fill || !p->tokens)
    return NULL;
  Token *token = p->tokens[p->mark];
  const char *text = lython_token_text(token);
  if (!token || token->type != NAME || !text || strcmp(text, keyword) != 0)
    return NULL;
  ++p->mark;
  return lython_ast_name_from_token(token);
}

expr_ty _PyPegen_soft_keyword_token(Parser *p) {
  if (!p || p->mark >= p->fill || !p->tokens)
    return NULL;
  Token *token = p->tokens[p->mark];
  const char *text = lython_token_text(token);
  if (!token || (token->type != NAME && token->type != SOFT_KEYWORD) ||
      !lython_is_soft_keyword_text(text))
    return NULL;
  ++p->mark;
  return lython_ast_name_from_token(token);
}

expr_ty _PyPegen_fstring_middle_token(Parser *p) {
  return (expr_ty)_PyPegen_expect_token(p, FSTRING_MIDDLE);
}

asdl_stmt_seq *_PyPegen_interactive_exit(Parser *p) {
  return (asdl_stmt_seq *)asdl_generic_seq_new(0, p ? p->arena : NULL);
}

Token *_PyPegen_get_last_nonnwhitespace_token(Parser *p) {
  if (!p || !p->tokens || p->mark <= 0)
    return NULL;
  return p->tokens[p->mark - 1];
}

int _PyPegen_fill_token(Parser *p) {
  if (!p || p->fill >= p->size) {
    lython_last_error_source = "_PyPegen_fill_token";
    return -1;
  }
  ++p->fill;
  return 0;
}

expr_ty _PyPegen_name_token(Parser *p) {
  Token *token = _PyPegen_expect_token(p, NAME);
  return lython_ast_name_from_token(token);
}

expr_ty _PyPegen_number_token(Parser *p) {
  return (expr_ty)_PyPegen_expect_token(p, NUMBER);
}

void *_PyPegen_string_token(Parser *p) {
  return _PyPegen_expect_token(p, STRING);
}

PyObject *_PyPegen_set_source_in_metadata(Parser *p, Token *t) {
  UNUSED(p);
  return t ? t->metadata : NULL;
}

void *_PyPegen_raise_error(Parser *p, PyObject *errtype, int use_mark,
                           const char *errmsg, ...) {
  UNUSED(errtype);
  UNUSED(use_mark);
  UNUSED(errmsg);
  if (p)
    p->error_indicator = 1;
  lython_last_error_source = "_PyPegen_raise_error";
  return NULL;
}

void *_PyPegen_raise_error_known_location(Parser *p, PyObject *errtype,
                                          int lineno, int col_offset,
                                          int end_lineno, int end_col_offset,
                                          const char *errmsg, ...) {
  UNUSED(lineno);
  UNUSED(col_offset);
  UNUSED(end_lineno);
  UNUSED(end_col_offset);
  return _PyPegen_raise_error(p, errtype, 0, errmsg);
}

void _Pypegen_set_syntax_error(Parser *p, Token *last_token) {
  UNUSED(last_token);
  if (p)
    p->error_indicator = 1;
  lython_last_error_source = "_Pypegen_set_syntax_error";
}

void _Pypegen_stack_overflow(Parser *p) {
  if (p)
    p->error_indicator = 1;
  lython_last_error_source = "_Pypegen_stack_overflow";
}

void *_PyPegen_dummy_name(Parser *p, ...) {
  UNUSED(p);
  return lython_cpython_ast_stub("dummy");
}

void *_PyPegen_seq_last_item(asdl_seq *seq) {
  if (!seq || seq->size == 0)
    return NULL;
  return seq->elements[seq->size - 1];
}

void *_PyPegen_seq_first_item(asdl_seq *seq) {
  if (!seq || seq->size == 0)
    return NULL;
  return seq->elements[0];
}

PyObject *_PyPegen_new_type_comment(Parser *p, const char *s) {
  UNUSED(p);
  return (PyObject *)s;
}

arg_ty _PyPegen_add_type_comment_to_arg(Parser *p, arg_ty arg, Token *tc) {
  UNUSED(p);
  UNUSED(tc);
  return arg;
}

PyObject *_PyPegen_new_identifier(Parser *p, const char *n) {
  UNUSED(p);
  return (PyObject *)n;
}

asdl_seq *_PyPegen_singleton_seq(Parser *p, void *a) {
  asdl_seq *seq = asdl_generic_seq_new(1, p ? p->arena : NULL);
  if (seq)
    seq->elements[0] = a;
  return seq;
}

asdl_seq *_PyPegen_seq_insert_in_front(Parser *p, void *a, asdl_seq *seq) {
  Py_ssize_t old_size = asdl_seq_LEN(seq);
  asdl_seq *result = asdl_generic_seq_new(old_size + 1, p ? p->arena : NULL);
  if (!result)
    return NULL;
  result->elements[0] = a;
  for (Py_ssize_t i = 0; i < old_size; ++i)
    result->elements[i + 1] = seq->elements[i];
  return result;
}

asdl_seq *_PyPegen_seq_append_to_end(Parser *p, asdl_seq *seq, void *a) {
  Py_ssize_t old_size = asdl_seq_LEN(seq);
  asdl_seq *result = asdl_generic_seq_new(old_size + 1, p ? p->arena : NULL);
  if (!result)
    return NULL;
  for (Py_ssize_t i = 0; i < old_size; ++i)
    result->elements[i] = seq->elements[i];
  result->elements[old_size] = a;
  return result;
}

asdl_seq *_PyPegen_seq_flatten(Parser *p, asdl_seq *seq) {
  if (!seq)
    return NULL;

  Py_ssize_t flat_size = 0;
  for (Py_ssize_t i = 0; i < seq->size; ++i) {
    void *element = seq->elements[i];
    if (lython_is_registered_asdl_seq(element))
      flat_size += ((asdl_seq *)element)->size;
    else if (element)
      ++flat_size;
  }

  asdl_seq *result = asdl_generic_seq_new(flat_size, p ? p->arena : NULL);
  if (!result)
    return NULL;

  Py_ssize_t out = 0;
  for (Py_ssize_t i = 0; i < seq->size; ++i) {
    void *element = seq->elements[i];
    if (lython_is_registered_asdl_seq(element)) {
      asdl_seq *nested = (asdl_seq *)element;
      for (Py_ssize_t j = 0; j < nested->size; ++j)
        result->elements[out++] = nested->elements[j];
      continue;
    }
    if (element)
      result->elements[out++] = element;
  }
  return result;
}

expr_ty _PyPegen_join_names_with_dot(Parser *p, expr_ty first, expr_ty second) {
  UNUSED(p);
  UNUSED(second);
  return first;
}

int _PyPegen_seq_count_dots(asdl_seq *seq) { return (int)asdl_seq_LEN(seq); }

alias_ty _PyPegen_alias_for_star(Parser *p, int lineno, int col_offset,
                                 int end_lineno, int end_col_offset,
                                 PyArena *arena) {
  UNUSED(p);
  UNUSED(lineno);
  UNUSED(col_offset);
  UNUSED(end_lineno);
  UNUSED(end_col_offset);
  UNUSED(arena);
  return lython_cpython_ast_stub("alias");
}

#define RETURN_SEQ(name, seq)                                                  \
  asdl_##name##_seq *_PyPegen_get_##name##s(Parser *p, asdl_seq *seq) {        \
    UNUSED(p);                                                                 \
    return (asdl_##name##_seq *)seq;                                           \
  }

asdl_identifier_seq *_PyPegen_map_names_to_ids(Parser *p, asdl_expr_seq *seq) {
  UNUSED(p);
  return (asdl_identifier_seq *)seq;
}

CmpopExprPair *_PyPegen_cmpop_expr_pair(Parser *p, cmpop_ty cmpop,
                                        expr_ty expr) {
  UNUSED(p);
  CmpopExprPair *pair =
      (CmpopExprPair *)lython_pegen_alloc(sizeof(CmpopExprPair));
  if (pair) {
    pair->cmpop = cmpop;
    pair->expr = expr;
  }
  return pair;
}

asdl_int_seq *_PyPegen_get_cmpops(Parser *p, asdl_seq *seq) {
  UNUSED(p);
  return (asdl_int_seq *)seq;
}

asdl_expr_seq *_PyPegen_get_exprs(Parser *p, asdl_seq *seq) {
  if (!seq)
    return NULL;
  asdl_seq *result =
      asdl_generic_seq_new(asdl_seq_LEN(seq), p ? p->arena : NULL);
  if (!result)
    return NULL;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    CmpopExprPair *pair = (CmpopExprPair *)seq->elements[i];
    result->elements[i] = pair ? pair->expr : NULL;
  }
  return (asdl_expr_seq *)result;
}

expr_ty _PyPegen_set_expr_context(Parser *p, expr_ty expr,
                                  expr_context_ty ctx) {
  UNUSED(p);
  UNUSED(ctx);
  return expr;
}

KeyValuePair *_PyPegen_key_value_pair(Parser *p, expr_ty key, expr_ty value) {
  UNUSED(p);
  KeyValuePair *pair = (KeyValuePair *)lython_pegen_alloc(sizeof(KeyValuePair));
  if (pair) {
    pair->key = key;
    pair->value = value;
  }
  return pair;
}

asdl_expr_seq *_PyPegen_get_keys(Parser *p, asdl_seq *seq) {
  if (!seq)
    return NULL;
  asdl_seq *result =
      asdl_generic_seq_new(asdl_seq_LEN(seq), p ? p->arena : NULL);
  if (!result)
    return NULL;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    KeyValuePair *pair = (KeyValuePair *)seq->elements[i];
    result->elements[i] = pair ? pair->key : NULL;
  }
  return (asdl_expr_seq *)result;
}

asdl_expr_seq *_PyPegen_get_values(Parser *p, asdl_seq *seq) {
  if (!seq)
    return NULL;
  asdl_seq *result =
      asdl_generic_seq_new(asdl_seq_LEN(seq), p ? p->arena : NULL);
  if (!result)
    return NULL;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    KeyValuePair *pair = (KeyValuePair *)seq->elements[i];
    result->elements[i] = pair ? pair->value : NULL;
  }
  return (asdl_expr_seq *)result;
}

KeyPatternPair *_PyPegen_key_pattern_pair(Parser *p, expr_ty key,
                                          pattern_ty pattern) {
  UNUSED(p);
  KeyPatternPair *pair =
      (KeyPatternPair *)lython_pegen_alloc(sizeof(KeyPatternPair));
  if (pair) {
    pair->key = key;
    pair->pattern = pattern;
  }
  return pair;
}

asdl_expr_seq *_PyPegen_get_pattern_keys(Parser *p, asdl_seq *seq) {
  if (!seq)
    return NULL;
  asdl_seq *result =
      asdl_generic_seq_new(asdl_seq_LEN(seq), p ? p->arena : NULL);
  if (!result)
    return NULL;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    KeyPatternPair *pair = (KeyPatternPair *)seq->elements[i];
    result->elements[i] = pair ? pair->key : NULL;
  }
  return (asdl_expr_seq *)result;
}

asdl_pattern_seq *_PyPegen_get_patterns(Parser *p, asdl_seq *seq) {
  if (!seq)
    return NULL;
  asdl_seq *result =
      asdl_generic_seq_new(asdl_seq_LEN(seq), p ? p->arena : NULL);
  if (!result)
    return NULL;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    KeyPatternPair *pair = (KeyPatternPair *)seq->elements[i];
    result->elements[i] = pair ? pair->pattern : NULL;
  }
  return (asdl_pattern_seq *)result;
}

NameDefaultPair *_PyPegen_name_default_pair(Parser *p, arg_ty arg,
                                            expr_ty value, Token *tc) {
  UNUSED(p);
  NameDefaultPair *pair =
      (NameDefaultPair *)lython_pegen_alloc(sizeof(NameDefaultPair));
  if (pair) {
    pair->arg = arg;
    pair->value = value;
    pair->tc = tc;
  }
  return pair;
}

SlashWithDefault *_PyPegen_slash_with_default(Parser *p,
                                              asdl_arg_seq *plain_names,
                                              asdl_seq *names_with_defaults) {
  UNUSED(p);
  SlashWithDefault *value =
      (SlashWithDefault *)lython_pegen_alloc(sizeof(SlashWithDefault));
  if (value) {
    value->plain_names = plain_names;
    value->names_with_defaults = names_with_defaults;
  }
  return value;
}

StarEtc *_PyPegen_star_etc(Parser *p, arg_ty vararg, asdl_seq *kwonlyargs,
                           arg_ty kwarg) {
  UNUSED(p);
  StarEtc *value = (StarEtc *)lython_pegen_alloc(sizeof(StarEtc));
  if (value) {
    value->vararg = vararg;
    value->kwonlyargs = kwonlyargs;
    value->kwarg = kwarg;
  }
  return value;
}

static Py_ssize_t lython_count_nonnull_seq(asdl_seq *seq) {
  Py_ssize_t count = 0;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    if (seq->elements[i])
      ++count;
  }
  return count;
}

static Py_ssize_t lython_count_name_default_args(asdl_seq *seq) {
  Py_ssize_t count = 0;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    NameDefaultPair *pair = (NameDefaultPair *)seq->elements[i];
    if (pair && pair->arg)
      ++count;
  }
  return count;
}

static Py_ssize_t lython_count_name_default_values(asdl_seq *seq) {
  Py_ssize_t count = 0;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    NameDefaultPair *pair = (NameDefaultPair *)seq->elements[i];
    if (pair && pair->value)
      ++count;
  }
  return count;
}

static void lython_append_nonnull_seq(asdl_seq *target, Py_ssize_t *out,
                                      asdl_seq *source) {
  if (!target || !out)
    return;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(source); ++i) {
    if (source->elements[i])
      target->elements[(*out)++] = source->elements[i];
  }
}

static void lython_append_name_default_args(asdl_seq *target, Py_ssize_t *out,
                                            asdl_seq *source) {
  if (!target || !out)
    return;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(source); ++i) {
    NameDefaultPair *pair = (NameDefaultPair *)source->elements[i];
    if (pair && pair->arg)
      target->elements[(*out)++] = pair->arg;
  }
}

static void lython_append_name_default_values(asdl_seq *target, Py_ssize_t *out,
                                              asdl_seq *source) {
  if (!target || !out)
    return;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(source); ++i) {
    NameDefaultPair *pair = (NameDefaultPair *)source->elements[i];
    if (pair && pair->value)
      target->elements[(*out)++] = pair->value;
  }
}

static void lython_append_one(asdl_seq *target, Py_ssize_t *out, void *value) {
  if (target && out && value)
    target->elements[(*out)++] = value;
}

typedef struct {
  asdl_arg_seq *slash_without;
  SlashWithDefault *slash_with;
  asdl_arg_seq *plain_names;
  asdl_seq *names_with_defaults;
  StarEtc *star_etc;
} PegenArguments;

static Py_ssize_t pegen_arguments_child_count(const PegenArguments *args) {
  Py_ssize_t total = 0;
  total += lython_count_nonnull_seq((asdl_seq *)args->slash_without);
  if (args->slash_with) {
    total +=
        lython_count_nonnull_seq((asdl_seq *)args->slash_with->plain_names);
    total +=
        lython_count_name_default_args(args->slash_with->names_with_defaults);
  }
  total += lython_count_nonnull_seq((asdl_seq *)args->plain_names);
  total += lython_count_name_default_args(args->names_with_defaults);
  if (args->star_etc) {
    if (args->star_etc->vararg)
      ++total;
    total += lython_count_name_default_args(args->star_etc->kwonlyargs);
    total += lython_count_name_default_values(args->star_etc->kwonlyargs);
    if (args->star_etc->kwarg)
      ++total;
  }
  if (args->slash_with)
    total +=
        lython_count_name_default_values(args->slash_with->names_with_defaults);
  total += lython_count_name_default_values(args->names_with_defaults);
  return total;
}

static void pegen_arguments_append(asdl_seq *children,
                                   const PegenArguments *args) {
  Py_ssize_t out = 0;
  lython_append_nonnull_seq(children, &out, (asdl_seq *)args->slash_without);
  if (args->slash_with) {
    lython_append_nonnull_seq(children, &out,
                              (asdl_seq *)args->slash_with->plain_names);
    lython_append_name_default_args(children, &out,
                                    args->slash_with->names_with_defaults);
  }
  lython_append_nonnull_seq(children, &out, (asdl_seq *)args->plain_names);
  lython_append_name_default_args(children, &out, args->names_with_defaults);
  if (args->star_etc) {
    lython_append_one(children, &out, args->star_etc->vararg);
    lython_append_name_default_args(children, &out, args->star_etc->kwonlyargs);
    lython_append_name_default_values(children, &out,
                                      args->star_etc->kwonlyargs);
    lython_append_one(children, &out, args->star_etc->kwarg);
  }
  if (args->slash_with)
    lython_append_name_default_values(children, &out,
                                      args->slash_with->names_with_defaults);
  lython_append_name_default_values(children, &out, args->names_with_defaults);
}

arguments_ty _PyPegen_make_arguments(Parser *p, asdl_arg_seq *slash_without,
                                     SlashWithDefault *slash_with,
                                     asdl_arg_seq *plain_names,
                                     asdl_seq *names_with_defaults,
                                     StarEtc *star_etc) {
  PegenArguments args = {slash_without, slash_with, plain_names,
                         names_with_defaults, star_etc};
  asdl_seq *children = asdl_generic_seq_new(pegen_arguments_child_count(&args),
                                            p ? p->arena : NULL);
  if (!children)
    return NULL;
  pegen_arguments_append(children, &args);
  return lython_cpython_ast_sequence("arguments", children);
}

arguments_ty _PyPegen_empty_arguments(Parser *p) {
  UNUSED(p);
  return lython_cpython_ast_stub("arguments");
}

static asdl_seq *lython_flat_string_parts(Parser *p, asdl_seq *source,
                                          const char *container_kind);

static expr_ty lython_wrap_debug_string(Parser *p, const char *container_kind,
                                        Token *debug_expr, expr_ty value) {
  if (!debug_expr)
    return value;
  asdl_seq *children = asdl_generic_seq_new(2, p ? p->arena : NULL);
  if (!children)
    return value;
  children->elements[0] = lython_cpython_ast_stub("Constant");
  children->elements[1] = value;
  return (expr_ty)lython_cpython_ast_sequence(container_kind, children);
}

expr_ty _PyPegen_template_str(Parser *p, Token *a,
                              asdl_expr_seq *raw_expressions, Token *b) {
  UNUSED(a);
  UNUSED(b);
  return (expr_ty)lython_cpython_ast_sequence(
      "TemplateStr",
      lython_flat_string_parts(p, (asdl_seq *)raw_expressions, "TemplateStr"));
}

expr_ty _PyPegen_joined_str(Parser *p, Token *a, asdl_expr_seq *raw_expressions,
                            Token *b) {
  UNUSED(a);
  UNUSED(b);
  return (expr_ty)lython_cpython_ast_sequence(
      "JoinedStr",
      lython_flat_string_parts(p, (asdl_seq *)raw_expressions, "JoinedStr"));
}

expr_ty _PyPegen_interpolation(Parser *p, expr_ty a, Token *b,
                               ResultTokenWithMetadata *c,
                               ResultTokenWithMetadata *d, Token *e, int l,
                               int c0, int el, int ec, PyArena *arena) {
  UNUSED(b);
  UNUSED(c);
  UNUSED(e);
  UNUSED(l);
  UNUSED(c0);
  UNUSED(el);
  UNUSED(ec);
  UNUSED(arena);
  expr_ty format_spec = d ? d->value : NULL;
  expr_ty interpolation = (expr_ty)lython_cpython_ast_children(
      "Interpolation", 2, (void *)a, (void *)format_spec);
  return lython_wrap_debug_string(p, "TemplateStr", b, interpolation);
}

expr_ty _PyPegen_formatted_value(Parser *p, expr_ty a, Token *b,
                                 ResultTokenWithMetadata *c,
                                 ResultTokenWithMetadata *d, Token *e, int l,
                                 int c0, int el, int ec, PyArena *arena) {
  UNUSED(c);
  UNUSED(e);
  UNUSED(l);
  UNUSED(c0);
  UNUSED(el);
  UNUSED(ec);
  UNUSED(arena);
  expr_ty format_spec = d ? d->value : NULL;
  expr_ty formatted = (expr_ty)lython_cpython_ast_children(
      "FormattedValue", 2, (void *)a, (void *)format_spec);
  return lython_wrap_debug_string(p, "JoinedStr", b, formatted);
}

void *_PyPegen_augoperator(Parser *p, operator_ty type) {
  UNUSED(p);
  AugOperator *op = (AugOperator *)lython_pegen_alloc(sizeof(AugOperator));
  if (op)
    op->kind = type;
  return op;
}

stmt_ty _PyPegen_function_def_decorators(Parser *p, asdl_expr_seq *decorators,
                                         stmt_ty function_def) {
  UNUSED(p);
  lython_ast_append_sequence_children(function_def, (asdl_seq *)decorators);
  return function_def;
}

stmt_ty _PyPegen_class_def_decorators(Parser *p, asdl_expr_seq *decorators,
                                      stmt_ty class_def) {
  UNUSED(p);
  lython_ast_append_sequence_children(class_def, (asdl_seq *)decorators);
  return class_def;
}

KeywordOrStarred *_PyPegen_keyword_or_starred(Parser *p, void *element,
                                              int is_keyword) {
  UNUSED(p);
  KeywordOrStarred *value =
      (KeywordOrStarred *)lython_pegen_alloc(sizeof(KeywordOrStarred));
  if (value) {
    value->element = element;
    value->is_keyword = is_keyword;
  }
  return value;
}

static asdl_seq *lython_extract_keyword_or_starred(Parser *p, asdl_seq *seq,
                                                   int want_keyword) {
  if (!seq)
    return NULL;

  Py_ssize_t count = 0;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    KeywordOrStarred *entry = (KeywordOrStarred *)seq->elements[i];
    if (entry && entry->is_keyword == want_keyword)
      ++count;
  }

  asdl_seq *result = asdl_generic_seq_new(count, p ? p->arena : NULL);
  if (!result)
    return NULL;

  Py_ssize_t out = 0;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    KeywordOrStarred *entry = (KeywordOrStarred *)seq->elements[i];
    if (entry && entry->is_keyword == want_keyword)
      result->elements[out++] = entry->element;
  }
  return result;
}

asdl_expr_seq *_PyPegen_seq_extract_starred_exprs(Parser *p, asdl_seq *seq) {
  return (asdl_expr_seq *)lython_extract_keyword_or_starred(p, seq,
                                                            /*want_keyword=*/0);
}

asdl_keyword_seq *_PyPegen_seq_delete_starred_exprs(Parser *p, asdl_seq *seq) {
  return (asdl_keyword_seq *)lython_extract_keyword_or_starred(
      p, seq, /*want_keyword=*/1);
}

expr_ty _PyPegen_collect_call_seqs(Parser *p, asdl_expr_seq *a, asdl_seq *b,
                                   int lineno, int col_offset, int end_lineno,
                                   int end_col_offset, PyArena *arena) {
  UNUSED(lineno);
  UNUSED(col_offset);
  UNUSED(end_lineno);
  UNUSED(end_col_offset);
  UNUSED(arena);
  asdl_expr_seq *args = a;
  asdl_expr_seq *starred = _PyPegen_seq_extract_starred_exprs(p, b);
  if (starred && asdl_seq_LEN(starred) > 0)
    args = (asdl_expr_seq *)_PyPegen_join_sequences(p, (asdl_seq *)args,
                                                    (asdl_seq *)starred);
  asdl_keyword_seq *keywords = _PyPegen_seq_delete_starred_exprs(p, b);
  return lython_cpython_ast_call(NULL, args, keywords);
}

expr_ty _PyPegen_constant_from_token(Parser *p, Token *tok) {
  UNUSED(p);
  expr_ty node = (expr_ty)lython_cpython_ast_unary("Constant", tok);
  if (!node || !tok)
    return node;
  node->lineno = tok->lineno;
  node->col_offset = tok->col_offset;
  node->end_lineno = tok->end_lineno;
  node->end_col_offset = tok->end_col_offset;
  return node;
}

expr_ty _PyPegen_decoded_constant_from_token(Parser *p, Token *tok) {
  return _PyPegen_constant_from_token(p, tok);
}

expr_ty _PyPegen_constant_from_string(Parser *p, Token *tok) {
  return _PyPegen_constant_from_token(p, tok);
}

static asdl_seq *lython_string_container_seq(void *node,
                                             const char *container_kind) {
  const struct _expr *ast = lython_registered_ast_node(node);
  if (!ast || !ast->lython_kind ||
      strcmp(ast->lython_kind, container_kind) != 0)
    return NULL;
  return ast->lython_seq;
}

static void lython_count_flat_string_parts(asdl_seq *seq,
                                           const char *container_kind,
                                           Py_ssize_t *count,
                                           int *previous_constant) {
  if (!seq)
    return;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    void *element = seq->elements[i];
    asdl_seq *nested = lython_string_container_seq(element, container_kind);
    if (nested) {
      lython_count_flat_string_parts(nested, container_kind, count,
                                     previous_constant);
      continue;
    }

    const char *kind = lython_registered_ast_kind(element);
    if (strcmp(kind, "Constant") == 0) {
      if (*previous_constant)
        continue;
      *previous_constant = 1;
    } else {
      *previous_constant = 0;
    }
    ++*count;
  }
}

static void lython_fill_flat_string_parts(asdl_seq *source,
                                          const char *container_kind,
                                          asdl_seq *target, Py_ssize_t *out,
                                          int *previous_constant) {
  if (!source || !target)
    return;
  for (Py_ssize_t i = 0; i < asdl_seq_LEN(source); ++i) {
    void *element = source->elements[i];
    asdl_seq *nested = lython_string_container_seq(element, container_kind);
    if (nested) {
      lython_fill_flat_string_parts(nested, container_kind, target, out,
                                    previous_constant);
      continue;
    }

    const char *kind = lython_registered_ast_kind(element);
    if (strcmp(kind, "Constant") == 0) {
      if (*previous_constant)
        continue;
      *previous_constant = 1;
    } else {
      *previous_constant = 0;
    }
    target->elements[(*out)++] = element;
  }
}

static asdl_seq *lython_flat_string_parts(Parser *p, asdl_seq *seq,
                                          const char *container_kind) {
  Py_ssize_t count = 0;
  int previous_constant = 0;
  lython_count_flat_string_parts(seq, container_kind, &count,
                                 &previous_constant);

  asdl_seq *result = asdl_generic_seq_new(count, p ? p->arena : NULL);
  if (!result)
    return NULL;

  Py_ssize_t out = 0;
  previous_constant = 0;
  lython_fill_flat_string_parts(seq, container_kind, result, &out,
                                &previous_constant);
  return result;
}

expr_ty _PyPegen_concatenate_tstrings(Parser *p, asdl_expr_seq *seq, int l,
                                      int c, int el, int ec, PyArena *arena) {
  UNUSED(l);
  UNUSED(c);
  UNUSED(el);
  UNUSED(ec);
  UNUSED(arena);
  return _PyPegen_template_str(p, NULL,
                               (asdl_expr_seq *)lython_flat_string_parts(
                                   p, (asdl_seq *)seq, "TemplateStr"),
                               NULL);
}

expr_ty _PyPegen_concatenate_strings(Parser *p, asdl_expr_seq *seq, int l,
                                     int c, int el, int ec, PyArena *arena) {
  UNUSED(l);
  UNUSED(c);
  UNUSED(el);
  UNUSED(ec);
  UNUSED(arena);
  if (asdl_seq_LEN(seq) == 1)
    return (expr_ty)asdl_seq_GET(seq, 0);

  for (Py_ssize_t i = 0; i < asdl_seq_LEN(seq); ++i) {
    const char *kind = lython_registered_ast_kind(asdl_seq_GET(seq, i));
    if (strcmp(kind, "JoinedStr") == 0 || strcmp(kind, "FormattedValue") == 0)
      return (expr_ty)lython_cpython_ast_sequence(
          "JoinedStr",
          lython_flat_string_parts(p, (asdl_seq *)seq, "JoinedStr"));
  }

  return _PyPegen_constant_from_token(p, NULL);
}

expr_ty _PyPegen_FetchRawForm(Parser *p, int l, int c, int el, int ec) {
  UNUSED(l);
  UNUSED(c);
  UNUSED(el);
  UNUSED(ec);
  return _PyPegen_constant_from_token(p, NULL);
}

expr_ty _PyPegen_ensure_imaginary(Parser *p, expr_ty e) {
  UNUSED(p);
  return e;
}

expr_ty _PyPegen_ensure_real(Parser *p, expr_ty e) {
  UNUSED(p);
  return e;
}

asdl_seq *_PyPegen_join_sequences(Parser *p, asdl_seq *a, asdl_seq *b) {
  UNUSED(p);
  if (!a)
    return b;
  if (!b)
    return a;
  asdl_seq *result =
      asdl_generic_seq_new(a->size + b->size, p ? p->arena : NULL);
  if (!result)
    return NULL;
  for (Py_ssize_t i = 0; i < a->size; ++i)
    result->elements[i] = a->elements[i];
  for (Py_ssize_t i = 0; i < b->size; ++i)
    result->elements[a->size + i] = b->elements[i];
  return result;
}

int _PyPegen_check_barry_as_flufl(Parser *p, Token *t) {
  UNUSED(p);
  UNUSED(t);
  return 0;
}

int _PyPegen_check_legacy_stmt(Parser *p, expr_ty t) {
  UNUSED(p);
  UNUSED(t);
  return 0;
}

ResultTokenWithMetadata *_PyPegen_check_fstring_conversion(Parser *p, Token *t,
                                                           expr_ty e) {
  UNUSED(p);
  ResultTokenWithMetadata *value =
      (ResultTokenWithMetadata *)lython_pegen_alloc(
          sizeof(ResultTokenWithMetadata));
  if (value) {
    value->value = e;
    value->metadata = t ? t->metadata : NULL;
  }
  return value;
}

ResultTokenWithMetadata *
_PyPegen_setup_full_format_spec(Parser *p, Token *t, asdl_expr_seq *seq, int l,
                                int c, int el, int ec, PyArena *arena) {
  UNUSED(l);
  UNUSED(c);
  UNUSED(el);
  UNUSED(ec);
  UNUSED(arena);
  ResultTokenWithMetadata *value =
      (ResultTokenWithMetadata *)lython_pegen_alloc(
          sizeof(ResultTokenWithMetadata));
  if (value) {
    value->value = (expr_ty)lython_cpython_ast_sequence(
        "JoinedStr", lython_flat_string_parts(p, (asdl_seq *)seq, "JoinedStr"));
    value->metadata = t ? t->metadata : NULL;
  }
  return value;
}

mod_ty _PyPegen_make_module(Parser *p, asdl_stmt_seq *stmts) {
  const Py_ssize_t stmt_count = asdl_seq_LEN((asdl_seq *)stmts);
  const Py_ssize_t type_ignore_count =
      p ? (Py_ssize_t)p->type_ignore_comments.num_items : 0;
  asdl_seq *children =
      asdl_generic_seq_new(stmt_count + type_ignore_count, p ? p->arena : NULL);
  if (!children)
    return NULL;
  Py_ssize_t out = 0;
  for (Py_ssize_t i = 0; i < stmt_count; ++i)
    children->elements[out++] = ((asdl_seq *)stmts)->elements[i];
  for (Py_ssize_t i = 0; i < type_ignore_count; ++i)
    children->elements[out++] = lython_cpython_ast_stub("TypeIgnore");
  return lython_cpython_ast_sequence("Module", children);
}

void *_PyPegen_arguments_parsing_error(Parser *p, expr_ty e) {
  UNUSED(e);
  return _PyPegen_raise_error(p, PyExc_SyntaxError, 0,
                              "invalid function arguments");
}

expr_ty _PyPegen_get_last_comprehension_item(comprehension_ty comprehension) {
  return (expr_ty)comprehension;
}

void *_PyPegen_nonparen_genexp_in_call(Parser *p, expr_ty args,
                                       asdl_comprehension_seq *comprehensions) {
  UNUSED(args);
  UNUSED(comprehensions);
  return _PyPegen_raise_error(p, PyExc_SyntaxError, 0,
                              "generator expression must be parenthesized");
}

stmt_ty _PyPegen_checked_future_import(Parser *p, identifier module,
                                       asdl_alias_seq *names, int lineno,
                                       int col_offset, int end_lineno,
                                       int end_col_offset, int feature_version,
                                       PyArena *arena) {
  UNUSED(p);
  UNUSED(module);
  UNUSED(lineno);
  UNUSED(col_offset);
  UNUSED(end_lineno);
  UNUSED(end_col_offset);
  UNUSED(feature_version);
  UNUSED(arena);
  return lython_cpython_ast_sequence("ImportFrom", (asdl_seq *)names);
}

asdl_stmt_seq *_PyPegen_register_stmts(Parser *p, asdl_stmt_seq *stmts) {
  UNUSED(p);
  return stmts;
}

stmt_ty _PyPegen_register_stmt(Parser *p, stmt_ty s) {
  UNUSED(p);
  return s;
}

expr_ty _PyPegen_get_invalid_target(expr_ty e, TARGETS_TYPE targets_type) {
  UNUSED(targets_type);
  return e;
}

const char *_PyPegen_get_expr_name(expr_ty e) {
  UNUSED(e);
  return "expression";
}
