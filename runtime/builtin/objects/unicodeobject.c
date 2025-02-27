/**
 * runtime/builtin/objects/unicodeobject.c
 * Lython Unicode文字列オブジェクトの実装
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <wchar.h>
#include <gc.h>
#include "object.h"
#include "unicodeobject.h"

/* Unicode型オブジェクト定義の前方宣言 */
static PyObject* unicode_repr(PyObject *op);
static PyObject* unicode_str(PyObject *op);
static Py_hash_t unicode_hash(PyObject *op);
static PyObject* unicode_richcompare(PyObject *a, PyObject *b, int op);
static void unicode_dealloc(PyObject *op);
static Py_ssize_t unicode_length(PyObject *op);
static PyObject* unicode_concat(PyObject *a, PyObject *b);
static PyObject* unicode_item(PyObject *op, Py_ssize_t i);

/* Unicode型の数値メソッド (空) */
static PyNumberMethods unicode_as_number = {
    0,                          /* nb_add */
    0,                          /* nb_subtract */
    0,                          /* nb_multiply */
    0,                          /* nb_remainder */
    0,                          /* nb_divmod */
    0,                          /* nb_power */
    0,                          /* nb_negative */
    0,                          /* nb_positive */
    0,                          /* nb_absolute */
};

/* Unicode型のシーケンスメソッド */
static PySequenceMethods unicode_as_sequence = {
    unicode_length,             /* sq_length */
    unicode_concat,             /* sq_concat */
    0,                          /* sq_repeat */
    unicode_item,               /* sq_item */
};

/* Unicode型オブジェクト定義 */
PyTypeObject PyUnicode_Type = {
    {1, NULL},                   /* PyObject_HEAD */
    "str",                       /* tp_name */
    sizeof(PyUnicodeObject),     /* tp_basicsize */
    0,                           /* tp_itemsize */
    unicode_dealloc,             /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    unicode_repr,                /* tp_repr */
    &unicode_as_number,          /* tp_as_number */
    &unicode_as_sequence,        /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    unicode_hash,                /* tp_hash */
    0,                           /* tp_call */
    unicode_str,                 /* tp_str */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    unicode_richcompare,         /* tp_richcompare */
    0,                           /* tp_dict */
    0,                           /* tp_base */
    0,                           /* tp_bases */
    0,                           /* tp_new */
    0,                           /* tp_init */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
};

/**
 * Unicode文字列オブジェクトのデストラクタ
 * (GCにより実際にはほとんど使われない)
 */
static void unicode_dealloc(PyObject *op) {
    PyUnicodeObject *unicode = (PyUnicodeObject *)op;
    
    /* GCがメモリを管理するので明示的解放は不要 */
    /* ただし参照が解放されるタイミングでフックできるようにしておく */
    
    /* 型を取り除いて循環参照を防ぐ */
    Py_TYPE(op) = NULL;
}

/**
 * Unicode文字列の文字数を返す
 */
static Py_ssize_t unicode_length(PyObject *op) {
    PyUnicodeObject *unicode = (PyUnicodeObject *)op;
    return unicode->length;
}

/**
 * Unicode文字列オブジェクトの表現を返す (__repr__)
 * 文字列をクォートで囲んで返す
 */
static PyObject* unicode_repr(PyObject *op) {
    PyUnicodeObject *unicode = (PyUnicodeObject *)op;
    const char *str = unicode->str;
    Py_ssize_t len = unicode->length;
    
    /* クォート付きの文字列を作成するためのバッファを確保 */
    /* 最大で len*4 + 2 (クォート) + 1 (NULL終端) のサイズになる可能性がある */
    Py_ssize_t max_repr_size = len * 4 + 3;
    char *repr_buf = (char *)GC_malloc(max_repr_size);
    if (repr_buf == NULL) {
        return NULL;
    }
    
    /* シングルクォートで囲む */
    repr_buf[0] = '\'';
    
    /* 文字列をエスケープしながらコピー */
    Py_ssize_t repr_len = 1;
    for (Py_ssize_t i = 0; i < len; i++) {
        unsigned char ch = (unsigned char)str[i];
        
        /* 特殊文字のエスケープ処理 */
        if (ch == '\\' || ch == '\'' || ch == '\t' || ch == '\n' || ch == '\r') {
            repr_buf[repr_len++] = '\\';
            switch (ch) {
                case '\\': repr_buf[repr_len++] = '\\'; break;
                case '\'': repr_buf[repr_len++] = '\''; break;
                case '\t': repr_buf[repr_len++] = 't'; break;
                case '\n': repr_buf[repr_len++] = 'n'; break;
                case '\r': repr_buf[repr_len++] = 'r'; break;
            }
        }
        /* 表示可能な文字はそのままコピー */
        else if (ch >= 32 && ch < 127) {
            repr_buf[repr_len++] = ch;
        }
        /* 非表示文字は16進エスケープシーケンスに変換 */
        else {
            repr_buf[repr_len++] = '\\';
            repr_buf[repr_len++] = 'x';
            repr_buf[repr_len++] = "0123456789abcdef"[(ch >> 4) & 0xf];
            repr_buf[repr_len++] = "0123456789abcdef"[ch & 0xf];
        }
    }
    
    /* 終端のクォートと NULL 文字 */
    repr_buf[repr_len++] = '\'';
    repr_buf[repr_len] = '\0';
    
    /* 新しいUnicode文字列オブジェクトを作成 */
    PyObject *result = PyUnicode_FromStringAndSize(repr_buf, repr_len);
    
    /* CPythonでは解放が必要だが、GCを使うなら不要 */
    /* GC_free(repr_buf); */
    
    return result;
}

/**
 * Unicode文字列オブジェクトの文字列表現を返す (__str__)
 * 自分自身を返すだけ
 */
static PyObject* unicode_str(PyObject *op) {
    Py_INCREF(op);
    return op;
}

/**
 * Unicode文字列のハッシュ値を計算
 * CPythonのMurmurHashアルゴリズムを簡略化したもの
 */
static Py_hash_t unicode_hash(PyObject *op) {
    PyUnicodeObject *unicode = (PyUnicodeObject *)op;
    
    /* ハッシュが既に計算済みならキャッシュを返す */
    if (unicode->hash != -1) {
        return unicode->hash;
    }
    
    /* ハッシュ計算 (シンプルなDJB2アルゴリズムを使用) */
    Py_hash_t hash = 5381;
    const char *str = unicode->str;
    Py_ssize_t len = unicode->length;
    
    for (Py_ssize_t i = 0; i < len; i++) {
        hash = ((hash << 5) + hash) + (unsigned char)str[i]; /* hash * 33 + c */
    }
    
    /* -1 はハッシュ未計算を表すので、-1 になった場合は -2 に調整 */
    if (hash == -1) {
        hash = -2;
    }
    
    /* ハッシュをキャッシュして返す */
    unicode->hash = hash;
    return hash;
}

/**
 * Unicode文字列の比較
 */
static PyObject* unicode_richcompare(PyObject *a, PyObject *b, int op) {
    /* bがUnicodeでなければNoneを返す(Python側で処理させる) */
    if (!PyUnicode_Check(b)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    PyUnicodeObject *unicode_a = (PyUnicodeObject *)a;
    PyUnicodeObject *unicode_b = (PyUnicodeObject *)b;
    
    /* 文字列比較 */
    int cmp = strcmp(unicode_a->str, unicode_b->str);
    int result;
    
    /* 比較演算子ごとに結果を評価 */
    switch (op) {
        case Py_LT: result = cmp < 0; break;
        case Py_LE: result = cmp <= 0; break;
        case Py_EQ: result = cmp == 0; break;
        case Py_NE: result = cmp != 0; break;
        case Py_GT: result = cmp > 0; break;
        case Py_GE: result = cmp >= 0; break;
        default: return NULL; /* 不正な演算子 */
    }
    
    /* 実際にはTrue/Falseオブジェクトを返す必要がある */
    if (result) {
        Py_INCREF(Py_True);
        return Py_True;
    } else {
        Py_INCREF(Py_False);
        return Py_False;
    }
}

/**
 * Unicode文字列の連結
 */
static PyObject* unicode_concat(PyObject *a, PyObject *b) {
    /* bがUnicodeでなければNone (未実装) */
    if (!PyUnicode_Check(b)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    PyUnicodeObject *unicode_a = (PyUnicodeObject *)a;
    PyUnicodeObject *unicode_b = (PyUnicodeObject *)b;
    
    /* 連結した文字列の長さを計算 */
    Py_ssize_t len_a = unicode_a->length;
    Py_ssize_t len_b = unicode_b->length;
    Py_ssize_t total_len = len_a + len_b;
    
    /* バッファを確保して文字列を連結 */
    char *result_buf = (char *)GC_malloc(total_len + 1);
    if (result_buf == NULL) {
        return NULL;
    }
    
    /* 文字列をコピー */
    memcpy(result_buf, unicode_a->str, len_a);
    memcpy(result_buf + len_a, unicode_b->str, len_b);
    result_buf[total_len] = '\0'; /* NULL終端 */
    
    /* 新しいUnicode文字列オブジェクトを作成 */
    return PyUnicode_FromStringAndSize(result_buf, total_len);
}

/**
 * Unicode文字列の指定位置の文字を取得
 */
static PyObject* unicode_item(PyObject *op, Py_ssize_t i) {
    PyUnicodeObject *unicode = (PyUnicodeObject *)op;
    
    /* インデックスが範囲外の場合 */
    if (i < 0 || i >= unicode->length) {
        /* IndexErrorを発生させるべきだが、ここではNULLを返す */
        return NULL;
    }
    
    /* 1文字分の文字列を作成 */
    char ch = unicode->str[i];
    return PyUnicode_FromStringAndSize(&ch, 1);
}

/**
 * 新しいUnicode文字列オブジェクトを作成 (NUL終端文字列から)
 */
PyObject* PyUnicode_FromString(const char *u) {
    if (u == NULL) {
        return NULL;
    }
    return PyUnicode_FromStringAndSize(u, strlen(u));
}

/**
 * 指定された長さの文字列からUnicode文字列オブジェクトを作成
 */
PyObject* PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size) {
    /* NULL文字列は空文字列として扱う */
    if (u == NULL && size == 0) {
        u = "";
    } else if (u == NULL) {
        return NULL;
    }
    
    /* Unicode文字列オブジェクト用メモリの確保 */
    PyUnicodeObject *unicode = (PyUnicodeObject *)GC_malloc(sizeof(PyUnicodeObject));
    if (unicode == NULL) {
        return NULL;
    }
    
    /* 文字列データ用のメモリを確保してコピー */
    char *str = (char *)GC_malloc(size + 1);
    if (str == NULL) {
        return NULL;
    }
    
    /* 文字列をコピーしてNULL終端 */
    memcpy(str, u, size);
    str[size] = '\0';
    
    /* Unicodeオブジェクトの初期化 */
    Py_TYPE(unicode) = &PyUnicode_Type;
    Py_REFCNT(unicode) = 1;
    unicode->length = size;
    unicode->hash = -1;  /* ハッシュ未計算 */
    unicode->kind = PyUnicode_WCHAR_KIND;
    unicode->str = str;
    
    return (PyObject *)unicode;
}

/**
 * フォーマット文字列からUnicode文字列オブジェクトを作成
 */
PyObject* PyUnicode_FromFormat(const char *format, ...) {
    va_list va;
    char buffer[1024];  /* 固定サイズバッファ (実際はもっと工夫が必要) */
    
    va_start(va, format);
    int len = vsnprintf(buffer, sizeof(buffer), format, va);
    va_end(va);
    
    if (len < 0 || len >= (int)sizeof(buffer)) {
        /* バッファオーバーフロー (対応が必要) */
        return NULL;
    }
    
    return PyUnicode_FromStringAndSize(buffer, len);
}

/**
 * 二つのUnicode文字列を連結
 */
PyObject* PyUnicode_Concat(PyObject *left, PyObject *right) {
    /* 両方がUnicode文字列であることを確認 */
    if (!PyUnicode_Check(left) || !PyUnicode_Check(right)) {
        return NULL;
    }
    
    /* unicode_concat関数を呼び出す */
    return unicode_concat(left, right);
}

/**
 * ワイド文字列からUnicode文字列オブジェクトを作成
 */
PyObject* PyUnicode_FromWideChar(const wchar_t *w, Py_ssize_t size) {
    if (w == NULL && size == 0) {
        return PyUnicode_FromStringAndSize("", 0);
    } else if (w == NULL) {
        return NULL;
    }
    
    /* サイズが指定されていない場合は自動計算 */
    if (size < 0) {
        size = wcslen(w);
    }
    
    /* マルチバイト変換用のバッファを確保 (最大サイズは要改善) */
    char *str = (char *)GC_malloc(size * 4 + 1);
    if (str == NULL) {
        return NULL;
    }
    
    /* ワイド文字をマルチバイト文字に変換 (シンプル化) */
    /* 実際にはUTF-8への正確な変換が必要 */
    Py_ssize_t len = wcstombs(str, w, size * 4);
    if (len == (size_t)-1) {
        /* 変換エラー */
        return NULL;
    }
    
    str[len] = '\0';  /* NULL終端 */
    
    return PyUnicode_FromStringAndSize(str, len);
}

/**
 * Unicode文字列の長さを取得
 */
Py_ssize_t PyUnicode_GetLength(PyObject *unicode) {
    if (!PyUnicode_Check(unicode)) {
        return -1;
    }
    return ((PyUnicodeObject *)unicode)->length;
}

/**
 * Unicode文字列のUTF-8表現を取得 (内部で長さも返す)
 */
const char* PyUnicode_AsUTF8AndSize(PyObject *unicode, Py_ssize_t *size) {
    if (!PyUnicode_Check(unicode)) {
        return NULL;
    }
    
    PyUnicodeObject *u = (PyUnicodeObject *)unicode;
    
    if (size != NULL) {
        *size = u->length;
    }
    
    return u->str;
}

/**
 * Unicode文字列のUTF-8表現を取得
 */
const char* PyUnicode_AsUTF8(PyObject *unicode) {
    return PyUnicode_AsUTF8AndSize(unicode, NULL);
}

/**
 * 二つのUnicode文字列を比較
 */
int PyUnicode_Compare(PyObject *left, PyObject *right) {
    /* 両方がUnicode文字列であることを確認 */
    if (!PyUnicode_Check(left) || !PyUnicode_Check(right)) {
        return -1;
    }
    
    PyUnicodeObject *unicode_left = (PyUnicodeObject *)left;
    PyUnicodeObject *unicode_right = (PyUnicodeObject *)right;
    
    return strcmp(unicode_left->str, unicode_right->str);
}

/**
 * Unicode文字列とASCII文字列を比較
 */
int PyUnicode_CompareWithASCIIString(PyObject *left, const char *right) {
    if (!PyUnicode_Check(left) || right == NULL) {
        return -1;
    }
    
    PyUnicodeObject *unicode_left = (PyUnicodeObject *)left;
    return strcmp(unicode_left->str, right);
}

/**
 * Unicode文字列のハッシュ値を計算 (外部インターフェイス)
 */
Py_hash_t PyUnicode_Hash(PyObject *unicode) {
    if (!PyUnicode_Check(unicode)) {
        return -1;
    }
    
    return unicode_hash(unicode);
}

/**
 * Unicode文字列サブシステムの初期化
 */
void _PyUnicode_Init(void) {
    /* 型オブジェクトの初期化 */
    PyType_Ready(&PyUnicode_Type);
    
    /* ここで空文字列など、よく使われる文字列の定数を初期化できる */
}
