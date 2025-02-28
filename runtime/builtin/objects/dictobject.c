/**
 * runtime/builtin/objects/dictobject.c
 * Lython辞書オブジェクトの実装
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gc.h>
#include "object.h"
#include "dictobject.h"
#include "listobject.h"
#include "unicodeobject.h"  /* 文字列表現のため */

/* 削除済みのマーカー */
static PyObject _dummy_struct = {
    1, NULL
};
#define dummy (&_dummy_struct)

/* 辞書型オブジェクト定義の前方宣言 */
static PyObject* dict_repr(PyObject *dict);
static PyObject* dict_str(PyObject *dict);
static Py_hash_t dict_hash(PyObject *dict);
static PyObject* dict_richcompare(PyObject *v, PyObject *w, int op);
static void dict_dealloc(PyObject *dict);
static Py_ssize_t dict_length(PyObject *dict);
static int dict_ass_subscript(PyObject *dict, PyObject *key, PyObject *value);
static PyObject* dict_subscript(PyObject *dict, PyObject *key);

/* 辞書型のマッピングメソッド */
static PyMappingMethods dict_as_mapping = {
    dict_length,             /* mp_length */
    dict_subscript,          /* mp_subscript */
    dict_ass_subscript,      /* mp_ass_subscript */
};

/* 辞書型オブジェクト定義 */
PyTypeObject PyDict_Type = {
    {1, NULL},                   /* PyObject_HEAD */
    "dict",                      /* tp_name */
    sizeof(PyDictObject),        /* tp_basicsize */
    0,                           /* tp_itemsize */
    dict_dealloc,                /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    dict_repr,                   /* tp_repr */
    0,                           /* tp_as_number */
    0,                           /* tp_as_sequence */
    &dict_as_mapping,            /* tp_as_mapping */
    dict_hash,                   /* tp_hash */
    0,                           /* tp_call */
    dict_str,                    /* tp_str */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    dict_richcompare,            /* tp_richcompare */
    0,                           /* tp_dict */
    0,                           /* tp_base */
    0,                           /* tp_bases */
    0,                           /* tp_new */
    0,                           /* tp_init */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
};

/**
 * エントリのLookupキー
 */
static PyDictEntry *lookdict(PyDictObject *dict, PyObject *key, Py_hash_t hash) {
    Py_ssize_t mask = dict->mask;
    Py_ssize_t i = hash & mask;
    PyDictEntry *entry = &dict->entries[i];
    PyDictEntry *dummy_entry = NULL;
    
    /* 空いているエントリか目的のキーを探す */
    while (entry->key != NULL) {
        if (entry->key == dummy) {
            /* 削除済み要素 */
            if (dummy_entry == NULL) {
                /* 最初に見つけた削除済みエントリを記録 */
                dummy_entry = entry;
            }
        }
        else if (entry->hash == hash) {
            /* ハッシュ値が一致 */
            PyObject *startkey = entry->key;
            /* キーを比較 */
            if (startkey == key) {
                return entry;
            }
            /* 値の比較 */
            int cmp = PyObject_RichCompareBool(startkey, key, Py_EQ);
            if (cmp > 0) {
                return entry;
            }
            else if (cmp < 0) {
                return NULL;  /* エラー発生 */
            }
        }
        
        /* 線形探索でインデックスを進める */
        i = (i + 1) & mask;
        entry = &dict->entries[i];
    }
    
    /* 見つからなかった場合 */
    return dummy_entry != NULL ? dummy_entry : entry;
}

/**
 * 辞書の拡張/縮小
 */
static int dictresize(PyDictObject *dict, Py_ssize_t newsize) {
    /* 少なくともPyDict_MINSIZEの大きさを確保 */
    if (newsize < PyDict_MINSIZE) {
        newsize = PyDict_MINSIZE;
    }
    
    /* 2のべき乗に切り上げ */
    Py_ssize_t size = 1;
    while (size < newsize) {
        size <<= 1;
    }
    
    /* 新しいエントリ配列を確保 */
    PyDictEntry *newtable = (PyDictEntry *)GC_malloc(size * sizeof(PyDictEntry));
    if (newtable == NULL) {
        return -1;
    }
    
    /* 新しいテーブルを初期化 */
    memset(newtable, 0, size * sizeof(PyDictEntry));
    
    /* 既存の要素を再ハッシュして新しいテーブルに入れる */
    Py_ssize_t mask = size - 1;
    PyDictEntry *oldentries = dict->entries;
    Py_ssize_t oldsize = dict->capacity;
    
    for (Py_ssize_t i = 0; i < oldsize; i++) {
        PyDictEntry *oldentry = &oldentries[i];
        
        /* キーがある場合だけ処理（削除済みマーカーも含む） */
        if (oldentry->key != NULL && oldentry->key != dummy) {
            /* キーのインデックスを計算 */
            Py_ssize_t index = oldentry->hash & mask;
            
            /* 空きスペースを探す */
            while (newtable[index].key != NULL) {
                index = (index + 1) & mask;
            }
            
            /* 要素を移動 */
            newtable[index] = *oldentry;
        }
    }
    
    /* 古いテーブルを解放して新しいテーブルを設定（GCの場合は明示的な解放は不要） */
    /* if (oldentries != NULL) {
        GC_free(oldentries);
    } */
    
    dict->entries = newtable;
    dict->capacity = size;
    dict->mask = mask;
    
    return 0;
}

/**
 * 辞書オブジェクトのデストラクタ
 * (GCにより実際にはほとんど使われない)
 */
static void dict_dealloc(PyObject *op) {
    PyDictObject *dict = (PyDictObject *)op;
    
    /* エントリの参照を解放 */
    if (dict->entries != NULL) {
        for (Py_ssize_t i = 0; i < dict->capacity; i++) {
            PyDictEntry *entry = &dict->entries[i];
            if (entry->key != NULL && entry->key != dummy) {
                Py_DECREF(entry->key);
                Py_DECREF(entry->value);
            }
        }
    }
    
    /* 型を取り除いて循環参照を防ぐ */
    Py_TYPE(op) = NULL;
}

/**
 * 辞書の要素数を返す
 */
static Py_ssize_t dict_length(PyObject *op) {
    PyDictObject *dict = (PyDictObject *)op;
    return dict->used;
}

/**
 * 辞書の文字列表現を生成
 */
static PyObject* dict_repr(PyObject *op) {
    PyDictObject *dict = (PyDictObject *)op;
    
    /* 空辞書の場合 */
    if (dict->used == 0) {
        return PyUnicode_FromString("{}");
    }
    
    /* 再帰的な処理を防ぐ (TODO) */
    
    /* バッファに文字列表現を構築 */
    /* バッファサイズは保守的に大きめに取る */
    char *buffer = (char *)GC_malloc(dict->used * 40 + 10);
    if (buffer == NULL) {
        return NULL;
    }
    
    /* 開始波括弧 */
    strcpy(buffer, "{");
    size_t pos = 1;
    
    /* 各要素の文字列表現を追加 */
    Py_ssize_t element_count = 0;
    
    for (Py_ssize_t i = 0; i < dict->capacity; i++) {
        PyDictEntry *entry = &dict->entries[i];
        if (entry->key != NULL && entry->key != dummy) {
            /* カンマと空白を追加（最初の要素以外） */
            if (element_count > 0) {
                strcpy(buffer + pos, ", ");
                pos += 2;
            }
            
            /* キーのrepr()を取得 */
            PyObject *key_repr = PyObject_Repr(entry->key);
            if (key_repr == NULL) {
                return NULL;
            }
            
            /* キーをバッファに追加 */
            const char *key_str = PyUnicode_AsUTF8(key_repr);
            if (key_str) {
                size_t len = strlen(key_str);
                strcpy(buffer + pos, key_str);
                pos += len;
            }
            
            Py_DECREF(key_repr);
            
            /* コロンと空白を追加 */
            strcpy(buffer + pos, ": ");
            pos += 2;
            
            /* 値のrepr()を取得 */
            PyObject *value_repr = PyObject_Repr(entry->value);
            if (value_repr == NULL) {
                return NULL;
            }
            
            /* 値をバッファに追加 */
            const char *value_str = PyUnicode_AsUTF8(value_repr);
            if (value_str) {
                size_t len = strlen(value_str);
                strcpy(buffer + pos, value_str);
                pos += len;
            }
            
            Py_DECREF(value_repr);
            
            element_count++;
        }
    }
    
    /* 終了波括弧 */
    strcpy(buffer + pos, "}");
    pos += 1;
    
    /* 最終的な文字列を作成 */
    PyObject *result = PyUnicode_FromStringAndSize(buffer, pos);
    
    /* GC_free(buffer); */
    
    return result;
}

/**
 * 辞書の文字列表現を生成 (str()と同じ)
 */
static PyObject* dict_str(PyObject *op) {
    return dict_repr(op);
}

/**
 * 辞書のハッシュ値を計算
 * 辞書はミュータブルなのでハッシュ不可
 */
static Py_hash_t dict_hash(PyObject *op) {
    /* ハッシュ不可を表す -1 を返す */
    return -1;
}

/**
 * 辞書の比較
 */
static PyObject* dict_richcompare(PyObject *v, PyObject *w, int op) {
    /* wが辞書でなければNoneを返す(Python側で処理させる) */
    if (!PyDict_Check(w)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    /* 等価性の比較のみサポート */
    if (op != Py_EQ && op != Py_NE) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    PyDictObject *dictv = (PyDictObject *)v;
    PyDictObject *dictw = (PyDictObject *)w;
    
    /* 要素数が違う場合 */
    if (dictv->used != dictw->used) {
        if (op == Py_EQ) {
            Py_INCREF(Py_False);
            return Py_False;
        } else {
            Py_INCREF(Py_True);
            return Py_True;
        }
    }
    
    /* 各キーと値を比較 */
    for (Py_ssize_t i = 0; i < dictv->capacity; i++) {
        PyDictEntry *entry = &dictv->entries[i];
        if (entry->key != NULL && entry->key != dummy) {
            /* wからキーに対応する値を取得 */
            PyObject *w_value = PyDict_GetItem(w, entry->key);
            if (w_value == NULL) {
                /* キーが存在しない */
                if (op == Py_EQ) {
                    Py_INCREF(Py_False);
                    return Py_False;
                } else {
                    Py_INCREF(Py_True);
                    return Py_True;
                }
            }
            
            /* 値を比較 */
            int cmp = PyObject_RichCompareBool(entry->value, w_value, Py_EQ);
            if (cmp < 0) {
                return NULL;  /* エラー */
            }
            if (cmp == 0) {
                /* 値が等しくない */
                if (op == Py_EQ) {
                    Py_INCREF(Py_False);
                    return Py_False;
                } else {
                    Py_INCREF(Py_True);
                    return Py_True;
                }
            }
        }
    }
    
    /* すべてのキーと値が一致 */
    if (op == Py_EQ) {
        Py_INCREF(Py_True);
        return Py_True;
    } else {
        Py_INCREF(Py_False);
        return Py_False;
    }
}

/**
 * 辞書からキーに対応する値を取得
 */
static PyObject* dict_subscript(PyObject *op, PyObject *key) {
    PyObject *value = PyDict_GetItem(op, key);
    if (value == NULL) {
        /* KeyError */
        return NULL;
    }
    
    Py_INCREF(value);
    return value;
}

/**
 * 辞書にキーと値のペアを設定または削除
 */
static int dict_ass_subscript(PyObject *op, PyObject *key, PyObject *value) {
    if (value == NULL) {
        /* 要素の削除 */
        return PyDict_DelItem(op, key);
    } else {
        /* 要素の設定 */
        return PyDict_SetItem(op, key, value);
    }
}

/**
 * 新しい辞書オブジェクトを作成
 */
PyObject* PyDict_New(void) {
    /* 辞書オブジェクトの割り当て */
    PyDictObject *dict = (PyDictObject *)GC_malloc(sizeof(PyDictObject));
    if (dict == NULL) {
        return NULL;
    }
    
    /* 初期化 */
    Py_TYPE(dict) = &PyDict_Type;
    Py_REFCNT(dict) = 1;
    dict->used = 0;
    dict->capacity = 0;
    dict->entries = NULL;
    dict->mask = 0;
    dict->hash = -1;  /* ハッシュ不可を表す -1 */
    
    /* 初期テーブルを確保 */
    if (dictresize(dict, PyDict_MINSIZE) < 0) {
        /* GC_free(dict); */
        return NULL;
    }
    
    return (PyObject *)dict;
}

/**
 * 辞書からキーに対応する値を取得
 */
PyObject* PyDict_GetItem(PyObject *dict, PyObject *key) {
    if (!PyDict_Check(dict)) {
        return NULL;
    }
    
    /* キーのハッシュ値を計算 */
    Py_hash_t hash = PyObject_Hash(key);
    if (hash == -1) {
        /* ハッシュ不可能なキー */
        return NULL;
    }
    
    PyDictObject *mp = (PyDictObject *)dict;
    PyDictEntry *entry = lookdict(mp, key, hash);
    
    if (entry == NULL || entry->key == NULL || entry->key == dummy) {
        return NULL;  /* キーが見つからない */
    }
    
    return entry->value;
}

/**
 * 辞書にキーと値のペアを設定
 */
int PyDict_SetItem(PyObject *dict, PyObject *key, PyObject *value) {
    if (!PyDict_Check(dict)) {
        return -1;
    }
    
    /* キーのハッシュ値を計算 */
    Py_hash_t hash = PyObject_Hash(key);
    if (hash == -1) {
        /* ハッシュ不可能なキー */
        return -1;
    }
    
    PyDictObject *mp = (PyDictObject *)dict;
    
    /* 容量が足りなければ拡張 */
    if (mp->used * 3 >= mp->capacity * 2) {
        if (dictresize(mp, mp->used * 2) < 0) {
            return -1;
        }
    }
    
    /* エントリを検索または新規作成 */
    PyDictEntry *entry = lookdict(mp, key, hash);
    if (entry == NULL) {
        return -1;
    }
    
    /* 既存のエントリか新規エントリか */
    int new_entry = entry->key == NULL || entry->key == dummy;
    
    /* 既存のエントリを置き換える場合 */
    if (!new_entry) {
        /* 古い値を解放 */
        Py_DECREF(entry->value);
    } else {
        /* 新しいエントリの場合 */
        if (entry->key == NULL) {
            /* キーの参照カウントを増やす */
            Py_INCREF(key);
            entry->key = key;
            entry->hash = hash;
            mp->used++;
        } else {
            /* 削除済みエントリを再利用 */
            Py_INCREF(key);
            entry->key = key;
            entry->hash = hash;
            mp->used++;
        }
    }
    
    /* 値を設定 */
    Py_INCREF(value);
    entry->value = value;
    
    return 0;
}

/**
 * 辞書からキーを削除
 */
int PyDict_DelItem(PyObject *dict, PyObject *key) {
    if (!PyDict_Check(dict)) {
        return -1;
    }
    
    /* キーのハッシュ値を計算 */
    Py_hash_t hash = PyObject_Hash(key);
    if (hash == -1) {
        /* ハッシュ不可能なキー */
        return -1;
    }
    
    PyDictObject *mp = (PyDictObject *)dict;
    PyDictEntry *entry = lookdict(mp, key, hash);
    
    /* キーが見つからない場合 */
    if (entry == NULL || entry->key == NULL || entry->key == dummy) {
        return -1;  /* KeyError */
    }
    
    /* キーと値の参照を解放 */
    Py_DECREF(entry->key);
    Py_DECREF(entry->value);
    
    /* エントリを削除済みマーカーで置き換え */
    entry->key = dummy;
    entry->value = NULL;
    
    mp->used--;
    
    /* 使用率が低い場合は縮小 */
    if (mp->capacity > PyDict_MINSIZE && mp->used * 10 < mp->capacity) {
        dictresize(mp, mp->capacity / 2);
    }
    
    return 0;
}

/**
 * 辞書の全要素を削除
 */
void PyDict_Clear(PyObject *dict) {
    if (!PyDict_Check(dict)) {
        return;
    }
    
    PyDictObject *mp = (PyDictObject *)dict;
    
    /* 各エントリの参照を解放 */
    for (Py_ssize_t i = 0; i < mp->capacity; i++) {
        PyDictEntry *entry = &mp->entries[i];
        if (entry->key != NULL && entry->key != dummy) {
            Py_DECREF(entry->key);
            Py_DECREF(entry->value);
            entry->key = NULL;
            entry->value = NULL;
        } else if (entry->key == dummy) {
            entry->key = NULL;
        }
    }
    
    mp->used = 0;
    
    /* テーブルを縮小 */
    if (mp->capacity > PyDict_MINSIZE) {
        dictresize(mp, PyDict_MINSIZE);
    }
}

/**
 * 辞書の反復処理のためのNext操作
 */
int PyDict_Next(PyObject *dict, Py_ssize_t *pos, PyObject **key, PyObject **value) {
    if (!PyDict_Check(dict)) {
        return 0;
    }
    
    PyDictObject *mp = (PyDictObject *)dict;
    Py_ssize_t i = *pos;
    
    /* 次の有効なエントリを探す */
    while (i < mp->capacity) {
        PyDictEntry *entry = &mp->entries[i];
        i++;
        if (entry->key != NULL && entry->key != dummy) {
            *pos = i;
            if (key) {
                *key = entry->key;
            }
            if (value) {
                *value = entry->value;
            }
            return 1;  /* 要素が見つかった */
        }
    }
    
    /* もう要素がない */
    *pos = i;
    return 0;
}

/**
 * 辞書の全キーをリストとして取得
 */
PyObject* PyDict_Keys(PyObject *dict) {
    if (!PyDict_Check(dict)) {
        return NULL;
    }
    
    PyDictObject *mp = (PyDictObject *)dict;
    
    /* 新しいリストを作成 */
    PyObject *list = PyList_New(mp->used);
    if (list == NULL) {
        return NULL;
    }
    
    /* キーをリストに追加 */
    Py_ssize_t index = 0;
    for (Py_ssize_t i = 0; i < mp->capacity; i++) {
        PyDictEntry *entry = &mp->entries[i];
        if (entry->key != NULL && entry->key != dummy) {
            Py_INCREF(entry->key);
            PyList_SetItem(list, index, entry->key);
            index++;
        }
    }
    
    return list;
}

/**
 * 辞書の全値をリストとして取得
 */
PyObject* PyDict_Values(PyObject *dict) {
    if (!PyDict_Check(dict)) {
        return NULL;
    }
    
    PyDictObject *mp = (PyDictObject *)dict;
    
    /* 新しいリストを作成 */
    PyObject *list = PyList_New(mp->used);
    if (list == NULL) {
        return NULL;
    }
    
    /* 値をリストに追加 */
    Py_ssize_t index = 0;
    for (Py_ssize_t i = 0; i < mp->capacity; i++) {
        PyDictEntry *entry = &mp->entries[i];
        if (entry->key != NULL && entry->key != dummy) {
            Py_INCREF(entry->value);
            PyList_SetItem(list, index, entry->value);
            index++;
        }
    }
    
    return list;
}

/**
 * 辞書の全キー・値ペアをリストとして取得
 */
PyObject* PyDict_Items(PyObject *dict) {
    if (!PyDict_Check(dict)) {
        return NULL;
    }
    
    PyDictObject *mp = (PyDictObject *)dict;
    
    /* 新しいリストを作成 */
    PyObject *list = PyList_New(mp->used);
    if (list == NULL) {
        return NULL;
    }
    
    /* キー・値ペアをリストに追加 */
    Py_ssize_t index = 0;
    for (Py_ssize_t i = 0; i < mp->capacity; i++) {
        PyDictEntry *entry = &mp->entries[i];
        if (entry->key != NULL && entry->key != dummy) {
            /* (key, value)のタプルを作成 */
            /* タプルがまだ未実装なので、仮にリストで代用 */
            PyObject *pair = PyList_New(2);
            if (pair == NULL) {
                Py_DECREF(list);
                return NULL;
            }
            
            Py_INCREF(entry->key);
            Py_INCREF(entry->value);
            PyList_SetItem(pair, 0, entry->key);
            PyList_SetItem(pair, 1, entry->value);
            
            PyList_SetItem(list, index, pair);
            index++;
        }
    }
    
    return list;
}

/**
 * 辞書の要素数を取得
 */
Py_ssize_t PyDict_Size(PyObject *dict) {
    if (!PyDict_Check(dict)) {
        return -1;
    }
    
    PyDictObject *mp = (PyDictObject *)dict;
    return mp->used;
}

/**
 * 辞書を別の辞書の内容で更新
 */
int PyDict_Update(PyObject *dict, PyObject *other) {
    return PyDict_Merge(dict, other, 1);
}

/**
 * 辞書を別の辞書の内容でマージ
 */
int PyDict_Merge(PyObject *dict, PyObject *other, int override) {
    if (!PyDict_Check(dict) || !PyDict_Check(other)) {
        return -1;
    }
    
    PyDictObject *mp_dict = (PyDictObject *)dict;
    PyDictObject *mp_other = (PyDictObject *)other;
    
    /* 他方の辞書のキーと値をコピー */
    for (Py_ssize_t i = 0; i < mp_other->capacity; i++) {
        PyDictEntry *entry = &mp_other->entries[i];
        if (entry->key != NULL && entry->key != dummy) {
            /* キーが既に存在し、上書きしない場合はスキップ */
            if (!override && PyDict_GetItem(dict, entry->key) != NULL) {
                continue;
            }
            
            /* キーと値をコピー */
            if (PyDict_SetItem(dict, entry->key, entry->value) < 0) {
                return -1;
            }
        }
    }
    
    return 0;
}

/**
 * 文字列キーから対応する値を取得
 */
PyObject* PyDict_GetItemString(PyObject *dict, const char *key) {
    if (!PyDict_Check(dict) || key == NULL) {
        return NULL;
    }
    
    /* 文字列キーをPyUnicodeObjectに変換 */
    PyObject *keyobj = PyUnicode_FromString(key);
    if (keyobj == NULL) {
        return NULL;
    }
    
    /* 通常のGetItem操作を実行 */
    PyObject *result = PyDict_GetItem(dict, keyobj);
    
    Py_DECREF(keyobj);
    return result;
}

/**
 * 文字列キーを使って値を設定
 */
int PyDict_SetItemString(PyObject *dict, const char *key, PyObject *value) {
    if (!PyDict_Check(dict) || key == NULL) {
        return -1;
    }
    
    /* 文字列キーをPyUnicodeObjectに変換 */
    PyObject *keyobj = PyUnicode_FromString(key);
    if (keyobj == NULL) {
        return -1;
    }
    
    /* 通常のSetItem操作を実行 */
    int result = PyDict_SetItem(dict, keyobj, value);
    
    Py_DECREF(keyobj);
    return result;
}

/**
 * 文字列キーを使って項目を削除
 */
int PyDict_DelItemString(PyObject *dict, const char *key) {
    if (!PyDict_Check(dict) || key == NULL) {
        return -1;
    }
    
    /* 文字列キーをPyUnicodeObjectに変換 */
    PyObject *keyobj = PyUnicode_FromString(key);
    if (keyobj == NULL) {
        return -1;
    }
    
    /* 通常のDelItem操作を実行 */
    int result = PyDict_DelItem(dict, keyobj);
    
    Py_DECREF(keyobj);
    return result;
}

/**
 * 辞書サブシステムの初期化
 */
void _PyDict_Init(void) {
    /* 型オブジェクトの初期化 */
    PyType_Ready(&PyDict_Type);
    
    /* ダミーオブジェクトの初期化 */
    _dummy_struct.ob_type = NULL;
}
