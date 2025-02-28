/**
 * runtime/builtin/objects/listobject.c
 * Lythonリストオブジェクトの実装
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gc.h>
#include "object.h"
#include "listobject.h"
#include "unicodeobject.h"  /* 文字列表現のため */

/* リスト型オブジェクト定義の前方宣言 */
static PyObject* list_repr(PyObject *v);
static PyObject* list_str(PyObject *v);
static Py_hash_t list_hash(PyObject *v);
static PyObject* list_richcompare(PyObject *v, PyObject *w, int op);
static void list_dealloc(PyObject *op);
static Py_ssize_t list_length(PyObject *op);
static PyObject* list_concat(PyObject *a, PyObject *b);
static PyObject* list_repeat(PyObject *a, Py_ssize_t n);
static PyObject* list_item(PyObject *op, Py_ssize_t i);
static int list_ass_item(PyObject *op, Py_ssize_t i, PyObject *v);
static int list_ass_slice(PyObject *a, Py_ssize_t ilow, Py_ssize_t ihigh, PyObject *v);

/* リスト型のシーケンスメソッド */
static PySequenceMethods list_as_sequence = {
    list_length,             /* sq_length */
    list_concat,             /* sq_concat */
    list_repeat,             /* sq_repeat */
    list_item,               /* sq_item */
    0,                       /* sq_slice (廃止) */
    list_ass_item,           /* sq_ass_item */
    list_ass_slice,          /* sq_ass_slice */
    0,                       /* sq_contains */
    0,                       /* sq_inplace_concat */
    0,                       /* sq_inplace_repeat */
};

/* リスト型オブジェクト定義 */
PyTypeObject PyList_Type = {
    {1, NULL},                   /* PyObject_HEAD */
    "list",                      /* tp_name */
    sizeof(PyListObject),        /* tp_basicsize */
    0,                           /* tp_itemsize */
    list_dealloc,                /* tp_dealloc */
    0,                           /* tp_print */
    0,                           /* tp_getattr */
    0,                           /* tp_setattr */
    list_repr,                   /* tp_repr */
    0,                           /* tp_as_number */
    &list_as_sequence,           /* tp_as_sequence */
    0,                           /* tp_as_mapping */
    list_hash,                   /* tp_hash */
    0,                           /* tp_call */
    list_str,                    /* tp_str */
    0,                           /* tp_iter */
    0,                           /* tp_iternext */
    list_richcompare,            /* tp_richcompare */
    0,                           /* tp_dict */
    0,                           /* tp_base */
    0,                           /* tp_bases */
    0,                           /* tp_new */
    0,                           /* tp_init */
    0,                           /* tp_traverse */
    0,                           /* tp_clear */
};

/**
 * リストオブジェクトのデストラクタ
 * (GCにより実際にはほとんど使われない)
 */
static void list_dealloc(PyObject *op) {
    PyListObject *self = (PyListObject *)op;
    Py_ssize_t i;
    
    /* 各要素の参照を減らす */
    for (i = 0; i < Py_SIZE(self); i++) {
        Py_XDECREF(self->ob_item[i]);
    }
    
    /* リスト自体の解放はGCに任せる */
    
    /* 型を取り除いて循環参照を防ぐ */
    Py_TYPE(op) = NULL;
}

/**
 * リストの文字数を返す
 */
static Py_ssize_t list_length(PyObject *op) {
    return Py_SIZE(op);
}

/**
 * リストの文字列表現を生成
 */
static PyObject* list_repr(PyObject *v) {
    PyListObject *self = (PyListObject *)v;
    Py_ssize_t i;
    
    /* 空リストの場合 */
    if (Py_SIZE(self) == 0) {
        return PyUnicode_FromString("[]");
    }
    
    /* 再帰的な処理を防ぐ (TODO) */
    
    /* バッファに文字列表現を構築 */
    /* バッファサイズは保守的に大きめに取る */
    char *buffer = (char *)GC_malloc(Py_SIZE(self) * 20 + 10);
    if (buffer == NULL) {
        return NULL;
    }
    
    /* 開始括弧 */
    strcpy(buffer, "[");
    size_t pos = 1;
    
    /* 各要素の文字列表現を追加 */
    for (i = 0; i < Py_SIZE(self); i++) {
        /* カンマと空白を追加（最初の要素以外） */
        if (i > 0) {
            strcpy(buffer + pos, ", ");
            pos += 2;
        }
        
        /* 要素のrepr()を取得 */
        PyObject *s = PyObject_Repr(self->ob_item[i]);
        if (s == NULL) {
            /* GC_free(buffer); */
            return NULL;
        }
        
        /* 文字列表現をバッファに追加 */
        const char *str = PyUnicode_AsUTF8(s);
        if (str) {
            size_t len = strlen(str);
            strcpy(buffer + pos, str);
            pos += len;
        }
        
        Py_DECREF(s);
    }
    
    /* 終了括弧 */
    strcpy(buffer + pos, "]");
    pos += 1;
    
    /* 最終的な文字列を作成 */
    PyObject *result = PyUnicode_FromStringAndSize(buffer, pos);
    
    /* GC_free(buffer); */
    
    return result;
}

/**
 * リストの文字列表現を生成 (str()と同じ)
 */
static PyObject* list_str(PyObject *v) {
    return list_repr(v);
}

/**
 * リストのハッシュ値を計算
 * リストはミュータブルなのでハッシュ不可
 */
static Py_hash_t list_hash(PyObject *v) {
    /* ハッシュ不可を表す -1 を返す */
    return -1;
}

/**
 * リストの比較
 */
static PyObject* list_richcompare(PyObject *v, PyObject *w, int op) {
    /* wがリストでなければNoneを返す(Python側で処理させる) */
    if (!PyList_Check(w)) {
        Py_INCREF(Py_None);
        return Py_None;
    }
    
    PyListObject *lv = (PyListObject *)v;
    PyListObject *lw = (PyListObject *)w;
    Py_ssize_t i;
    
    /* リストの長さが違う場合の簡易比較 */
    if (op == Py_EQ && Py_SIZE(lv) != Py_SIZE(lw)) {
        Py_INCREF(Py_False);
        return Py_False;
    }
    if (op == Py_NE && Py_SIZE(lv) != Py_SIZE(lw)) {
        Py_INCREF(Py_True);
        return Py_True;
    }
    
    /* 要素ごとに比較 */
    for (i = 0; i < Py_SIZE(lv) && i < Py_SIZE(lw); i++) {
        PyObject *res = PyObject_RichCompare(lv->ob_item[i], lw->ob_item[i], Py_EQ);
        int eq;
        
        if (res == NULL) {
            return NULL;
        }
        
        eq = PyObject_IsTrue(res);
        Py_DECREF(res);
        
        if (eq < 0) {
            return NULL;
        }
        
        if (!eq) {
            /* 要素が等しくない場合 */
            /* 詳細な比較を行う */
            res = PyObject_RichCompare(lv->ob_item[i], lw->ob_item[i], op);
            return res;
        }
    }
    
    /* ここまで来たら、共通部分は全て等しい */
    /* 長さの比較で決定 */
    Py_ssize_t vlen = Py_SIZE(lv);
    Py_ssize_t wlen = Py_SIZE(lw);
    int cmp;
    
    switch (op) {
        case Py_LT: cmp = vlen <  wlen; break;
        case Py_LE: cmp = vlen <= wlen; break;
        case Py_EQ: cmp = vlen == wlen; break;
        case Py_NE: cmp = vlen != wlen; break;
        case Py_GT: cmp = vlen >  wlen; break;
        case Py_GE: cmp = vlen >= wlen; break;
        default: return NULL; /* 不正な演算子 */
    }
    
    if (cmp) {
        Py_INCREF(Py_True);
        return Py_True;
    } else {
        Py_INCREF(Py_False);
        return Py_False;
    }
}

/**
 * リストの連結
 */
static PyObject* list_concat(PyObject *a, PyObject *b) {
    /* bがリストでなければNULL（未実装） */
    if (!PyList_Check(b)) {
        return NULL;
    }
    
    PyListObject *self = (PyListObject *)a;
    PyListObject *other = (PyListObject *)b;
    Py_ssize_t size = Py_SIZE(self) + Py_SIZE(other);
    
    /* 新しいリストを作成 */
    PyObject *result = PyList_New(size);
    if (result == NULL) {
        return NULL;
    }
    
    PyListObject *result_list = (PyListObject *)result;
    Py_ssize_t i;
    
    /* 自分自身の要素をコピー */
    for (i = 0; i < Py_SIZE(self); i++) {
        PyObject *item = self->ob_item[i];
        Py_INCREF(item);
        result_list->ob_item[i] = item;
    }
    
    /* 他方のリストの要素を追加 */
    for (i = 0; i < Py_SIZE(other); i++) {
        PyObject *item = other->ob_item[i];
        Py_INCREF(item);
        result_list->ob_item[Py_SIZE(self) + i] = item;
    }
    
    return result;
}

/**
 * リストの繰り返し
 */
static PyObject* list_repeat(PyObject *a, Py_ssize_t n) {
    PyListObject *self = (PyListObject *)a;
    Py_ssize_t size;
    Py_ssize_t i, j;
    
    /* 繰り返し回数が無効な場合 */
    if (n < 0) {
        n = 0;
    }
    
    /* オーバーフロー対策 */
    if (n && Py_SIZE(self) > (Py_ssize_t)-1 / n) {
        return NULL;  /* OverflowError */
    }
    
    /* 新しいリストのサイズを計算 */
    size = Py_SIZE(self) * n;
    
    /* 新しいリストを作成 */
    PyObject *result = PyList_New(size);
    if (result == NULL) {
        return NULL;
    }
    
    PyListObject *result_list = (PyListObject *)result;
    
    /* 要素をn回繰り返しコピー */
    for (i = 0; i < n; i++) {
        for (j = 0; j < Py_SIZE(self); j++) {
            PyObject *item = self->ob_item[j];
            Py_INCREF(item);
            result_list->ob_item[i * Py_SIZE(self) + j] = item;
        }
    }
    
    return result;
}

/**
 * リストの指定位置の要素を取得
 */
static PyObject* list_item(PyObject *op, Py_ssize_t i) {
    PyListObject *self = (PyListObject *)op;
    
    /* インデックスが範囲外の場合 */
    if (i < 0 || i >= Py_SIZE(self)) {
        /* IndexErrorを発生させるべきだが、ここではNULLを返す */
        return NULL;
    }
    
    /* 要素を取得して参照カウントを増やす */
    PyObject *item = self->ob_item[i];
    Py_INCREF(item);
    return item;
}

/**
 * リストの指定位置に要素を設定
 */
static int list_ass_item(PyObject *op, Py_ssize_t i, PyObject *v) {
    PyListObject *self = (PyListObject *)op;
    
    /* インデックスが範囲外の場合 */
    if (i < 0 || i >= Py_SIZE(self)) {
        return -1;  /* IndexError */
    }
    
    /* 要素の置き換え */
    if (v == NULL) {
        /* 削除操作 - リサイズが必要なのでスライス操作に委譲 */
        return list_ass_slice(op, i, i+1, NULL);
    } else {
        /* 既存の要素を置き換え */
        PyObject *old_item = self->ob_item[i];
        Py_INCREF(v);
        self->ob_item[i] = v;
        Py_XDECREF(old_item);
        return 0;
    }
}

/**
 * リストのスライスを置き換え・削除
 */
static int list_ass_slice(PyObject *a, Py_ssize_t ilow, Py_ssize_t ihigh, PyObject *v) {
    PyListObject *self = (PyListObject *)a;
    PyObject **item;
    Py_ssize_t i, n;
    
    /* 有効範囲に調整 */
    if (ilow < 0) {
        ilow = 0;
    } else if (ilow > Py_SIZE(self)) {
        ilow = Py_SIZE(self);
    }
    
    if (ihigh < ilow) {
        ihigh = ilow;
    } else if (ihigh > Py_SIZE(self)) {
        ihigh = Py_SIZE(self);
    }
    
    /* 置き換えられる要素数 */
    n = ihigh - ilow;
    
    /* 削除操作 */
    if (v == NULL) {
        /* 削除される要素の参照を減らす */
        for (i = 0; i < n; i++) {
            Py_XDECREF(self->ob_item[ilow + i]);
        }
        
        /* 残りの要素を前に移動 */
        memmove(&self->ob_item[ilow], &self->ob_item[ihigh],
                (Py_SIZE(self) - ihigh) * sizeof(PyObject *));
        
        /* サイズの調整 */
        Py_SET_SIZE(self, Py_SIZE(self) - n);
        return 0;
    }
    
    /* 挿入操作：vがリストではない場合 */
    if (!PyList_Check(v)) {
        return -1;  /* TypeError */
    }
    
    PyListObject *vlist = (PyListObject *)v;
    Py_ssize_t m = Py_SIZE(vlist);
    
    /* リストサイズの変更が必要な場合 */
    if (n != m) {
        /* 領域の拡張/縮小が必要 */
        Py_ssize_t new_size = Py_SIZE(self) + m - n;
        
        /* サイズ変更が必要なら容量を確保 */
        if (new_size > self->allocated) {
            /* 必要に応じて領域を拡張 */
            PyObject **items = (PyObject **)GC_malloc(new_size * sizeof(PyObject *));
            if (items == NULL) {
                return -1;  /* メモリエラー */
            }
            
            /* 前半部分をコピー */
            memcpy(items, self->ob_item, ilow * sizeof(PyObject *));
            
            /* 後半部分をコピー */
            memcpy(items + ilow + m, 
                   self->ob_item + ihigh, 
                   (Py_SIZE(self) - ihigh) * sizeof(PyObject *));
            
            /* 古い領域を解放（GCの場合は不要） */
            /* GC_free(self->ob_item); */
            
            /* 新しい領域を設定 */
            self->ob_item = items;
            self->allocated = new_size;
        } else if (m < n) {
            /* 縮小の場合、要素を前に詰める */
            memmove(&self->ob_item[ilow + m], 
                    &self->ob_item[ihigh],
                    (Py_SIZE(self) - ihigh) * sizeof(PyObject *));
        } else if (m > n) {
            /* 拡張の場合、要素を後ろにずらす */
            memmove(&self->ob_item[ilow + m], 
                    &self->ob_item[ihigh],
                    (Py_SIZE(self) - ihigh) * sizeof(PyObject *));
        }
        
        /* サイズを更新 */
        Py_SET_SIZE(self, new_size);
    }
    
    /* 削除される要素の参照を減らす */
    for (i = 0; i < n; i++) {
        Py_XDECREF(self->ob_item[ilow + i]);
    }
    
    /* 新しい要素をコピー */
    for (i = 0; i < m; i++) {
        PyObject *item = vlist->ob_item[i];
        Py_INCREF(item);
        self->ob_item[ilow + i] = item;
    }
    
    return 0;
}

/**
 * 新しいリストオブジェクトを作成
 */
PyObject* PyList_New(Py_ssize_t size) {
    /* サイズが無効な場合 */
    if (size < 0) {
        return NULL;
    }
    
    /* 割り当てサイズを調整 */
    Py_ssize_t allocated;
    if (size <= PyList_MINSIZE) {
        allocated = PyList_MINSIZE;
    } else {
        /* 少し余裕を持って確保 */
        allocated = size + (size >> 3) + (size < 9 ? 3 : 6);
    }
    
    /* リストオブジェクトの割り当て */
    PyListObject *op = (PyListObject *)GC_malloc(sizeof(PyListObject));
    if (op == NULL) {
        return NULL;
    }
    
    /* 要素配列の割り当て */
    PyObject **items = NULL;
    if (allocated > 0) {
        items = (PyObject **)GC_malloc(allocated * sizeof(PyObject *));
        if (items == NULL) {
            /* GC_free(op); */
            return NULL;
        }
    }
    
    /* リストの初期化 */
    Py_TYPE(op) = &PyList_Type;
    Py_REFCNT(op) = 1;
    Py_SET_SIZE(op, size);
    op->allocated = allocated;
    op->ob_item = items;
    
    /* 要素を初期化 */
    if (items != NULL) {
        memset(items, 0, allocated * sizeof(PyObject *));
    }
    
    return (PyObject *)op;
}

/**
 * リストのサイズを取得
 */
Py_ssize_t PyList_Size(PyObject *list) {
    if (!PyList_Check(list)) {
        return -1;
    }
    return Py_SIZE(list);
}

/**
 * リストの要素を取得
 */
PyObject* PyList_GetItem(PyObject *list, Py_ssize_t index) {
    if (!PyList_Check(list)) {
        return NULL;
    }
    
    PyListObject *self = (PyListObject *)list;
    
    /* インデックスが範囲外の場合 */
    if (index < 0 || index >= Py_SIZE(self)) {
        return NULL;
    }
    
    /* 参照カウントを増やさずに返す（高速アクセス用） */
    return self->ob_item[index];
}

/**
 * リストに要素を設定
 */
int PyList_SetItem(PyObject *list, Py_ssize_t index, PyObject *item) {
    if (!PyList_Check(list)) {
        Py_XDECREF(item);
        return -1;
    }
    
    PyListObject *self = (PyListObject *)list;
    
    /* インデックスが範囲外の場合 */
    if (index < 0 || index >= Py_SIZE(self)) {
        Py_XDECREF(item);
        return -1;
    }
    
    /* 既存の要素を解放して新しい要素を設定 */
    Py_XDECREF(self->ob_item[index]);
    self->ob_item[index] = item;  /* 所有権を移管するので参照カウント増加不要 */
    
    return 0;
}

/**
 * リストに要素を挿入
 */
int PyList_Insert(PyObject *list, Py_ssize_t index, PyObject *item) {
    if (!PyList_Check(list)) {
        return -1;
    }
    
    PyListObject *self = (PyListObject *)list;
    Py_ssize_t size = Py_SIZE(self);
    
    /* インデックスを有効範囲に調整 */
    if (index < 0) {
        index += size;
        if (index < 0) {
            index = 0;
        }
    }
    if (index > size) {
        index = size;
    }
    
    /* スライスの置き換えとして実装 */
    PyObject *temp = PyList_New(1);
    if (temp == NULL) {
        return -1;
    }
    
    PyList_SetItem(temp, 0, item);  /* 所有権を移管 */
    Py_INCREF(item);  /* SetItemで減少した分を補填 */
    
    int result = list_ass_slice(list, index, index, temp);
    Py_DECREF(temp);
    
    return result;
}

/**
 * リストに要素を追加
 */
int PyList_Append(PyObject *list, PyObject *item) {
    if (!PyList_Check(list)) {
        return -1;
    }
    
    PyListObject *self = (PyListObject *)list;
    Py_ssize_t size = Py_SIZE(self);
    
    /* 容量が足りなければ拡張 */
    if (size >= self->allocated) {
        /* 拡張した新しい配列を確保 */
        Py_ssize_t new_allocated = self->allocated + (self->allocated >> 1) + 1;
        PyObject **items = (PyObject **)GC_malloc(new_allocated * sizeof(PyObject *));
        if (items == NULL) {
            return -1;
        }
        
        /* 既存の要素をコピー */
        memcpy(items, self->ob_item, size * sizeof(PyObject *));
        
        /* 古い配列を解放（GCの場合は不要） */
        /* GC_free(self->ob_item); */
        
        /* 新しい配列を設定 */
        self->ob_item = items;
        self->allocated = new_allocated;
    }
    
    /* 要素を追加 */
    Py_INCREF(item);
    self->ob_item[size] = item;
    Py_SET_SIZE(self, size + 1);
    
    return 0;
}

/**
 * リストのスライスを取得
 */
PyObject* PyList_GetSlice(PyObject *list, Py_ssize_t low, Py_ssize_t high) {
    if (!PyList_Check(list)) {
        return NULL;
    }
    
    PyListObject *self = (PyListObject *)list;
    Py_ssize_t size = Py_SIZE(self);
    
    /* インデックスを有効範囲に調整 */
    if (low < 0) {
        low = 0;
    } else if (low > size) {
        low = size;
    }
    
    if (high < low) {
        high = low;
    } else if (high > size) {
        high = size;
    }
    
    /* スライスの長さ */
    Py_ssize_t slice_len = high - low;
    
    /* 新しいリストを作成 */
    PyObject *result = PyList_New(slice_len);
    if (result == NULL) {
        return NULL;
    }
    
    PyListObject *result_list = (PyListObject *)result;
    
    /* 要素をコピー */
    for (Py_ssize_t i = 0; i < slice_len; i++) {
        PyObject *item = self->ob_item[low + i];
        Py_INCREF(item);
        result_list->ob_item[i] = item;
    }
    
    return result;
}

/**
 * リストのスライスを設定
 */
int PyList_SetSlice(PyObject *list, Py_ssize_t low, Py_ssize_t high, PyObject *itemlist) {
    if (!PyList_Check(list)) {
        return -1;
    }
    
    /* スライス操作に委譲 */
    return list_ass_slice(list, low, high, itemlist);
}

/**
 * リストをソート
 */
int PyList_Sort(PyObject *list) {
    /* 単純な挿入ソートを実装（効率は良くない） */
    /* 実際には比較関数を使ったクイックソートなどが適切 */
    if (!PyList_Check(list)) {
        return -1;
    }
    
    PyListObject *self = (PyListObject *)list;
    Py_ssize_t size = Py_SIZE(self);
    
    /* 1つ以下なら何もしない */
    if (size <= 1) {
        return 0;
    }
    
    /* 挿入ソート */
    for (Py_ssize_t i = 1; i < size; i++) {
        PyObject *key = self->ob_item[i];
        Py_ssize_t j = i - 1;
        
        while (j >= 0) {
            /* j番目の要素とkeyを比較 */
            PyObject *res = PyObject_RichCompare(self->ob_item[j], key, Py_GT);
            int gt;
            
            if (res == NULL) {
                return -1;
            }
            
            gt = PyObject_IsTrue(res);
            Py_DECREF(res);
            
            if (gt < 0) {
                return -1;
            }
            
            if (!gt) {
                break;
            }
            
            /* j+1番目にj番目の値を移動 */
            self->ob_item[j + 1] = self->ob_item[j];
            j--;
        }
        
        /* j+1番目にkeyを挿入 */
        self->ob_item[j + 1] = key;
    }
    
    return 0;
}

/**
 * リストを反転
 */
int PyList_Reverse(PyObject *list) {
    if (!PyList_Check(list)) {
        return -1;
    }
    
    PyListObject *self = (PyListObject *)list;
    Py_ssize_t size = Py_SIZE(self);
    
    /* 要素が1つ以下なら何もしない */
    if (size <= 1) {
        return 0;
    }
    
    /* 要素を反転 */
    for (Py_ssize_t i = 0; i < size / 2; i++) {
        PyObject *temp = self->ob_item[i];
        self->ob_item[i] = self->ob_item[size - i - 1];
        self->ob_item[size - i - 1] = temp;
    }
    
    return 0;
}

/**
 * リストをタプルに変換
 */
PyObject* PyList_AsTuple(PyObject *list) {
    /* タプル型が未実装なので、現時点ではNULLを返す */
    return NULL;
}

/**
 * リストサブシステムの初期化
 */
void _PyList_Init(void) {
    /* 型オブジェクトの初期化 */
    PyType_Ready(&PyList_Type);
}
