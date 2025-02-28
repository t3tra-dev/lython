/**
 * runtime/builtin/objects/object.h
 * Lython基本オブジェクトシステムの定義
 * CPythonのオブジェクトシステムを参考に設計
 */

#ifndef LYTHON_OBJECT_H
#define LYTHON_OBJECT_H

#include <stddef.h>
#include <stdint.h>

/* 初期化関数 */
void PyObject_Init(void);
void PyObject_InitSystem(void);

#ifdef __cplusplus
extern "C" {
#endif

/* 前方宣言 */
typedef struct _typeobject PyTypeObject;

/* Py_ssize_t型の定義 (CPythonと互換) */
#ifdef _WIN64
typedef int64_t Py_ssize_t;
#else
typedef long Py_ssize_t;
#endif

/* Py_hash_t型の定義 (ハッシュ値用) */
typedef Py_ssize_t Py_hash_t;



/* 基本的なPyObjectの構造体 - すべてのオブジェクトの基底 */
typedef struct _object {
    Py_ssize_t ob_refcnt;       /* 参照カウント (GCサポート用) */
    PyTypeObject *ob_type;      /* オブジェクトの型 */
} PyObject;

/* 可変長オブジェクト用 */
typedef struct {
    PyObject ob_base;
    Py_ssize_t ob_size;         /* 可変長オブジェクトのサイズ */
} PyVarObject;

/* オブジェクトフィールドにアクセスするためのマクロ */
#define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)
#define Py_REFCNT(ob) (((PyObject*)(ob))->ob_refcnt)
#define Py_SIZE(ob) (((PyVarObject*)(ob))->ob_size)
#define Py_SET_SIZE(ob, size) ((((PyVarObject*)(ob))->ob_size) = (size))

/* Py_ssize_t型の定義 (CPythonと互換) */
#ifdef _WIN64
typedef int64_t Py_ssize_t;
#else
typedef long Py_ssize_t;
#endif

/* ブール値関連の関数 */
PyObject* PyBool_FromLong(long v);

/* 長さを返す汎用インターフェース */
typedef Py_ssize_t (*lenfunc)(PyObject *);

/* 単項演算子のインターフェース */
typedef PyObject *(*unaryfunc)(PyObject *);

/* 二項演算子のインターフェース */
typedef PyObject *(*binaryfunc)(PyObject *, PyObject *);

/* 三項演算子のインターフェース */
typedef PyObject *(*ternaryfunc)(PyObject *, PyObject *, PyObject *);

/* getattr/setattr用インターフェース */
typedef PyObject *(*getattrfunc)(PyObject *, char *);
typedef int (*setattrfunc)(PyObject *, char *, PyObject *);

/* リッチ比較のインターフェース */
typedef PyObject *(*richcmpfunc)(PyObject *, PyObject *, int);

/* リプレゼンテーション関数 */
typedef PyObject *(*reprfunc)(PyObject *);

/* ハッシュ関数 */
typedef Py_ssize_t (*hashfunc)(PyObject *);

/* コール関数 (callable) */
typedef PyObject *(*ternaryfunc)(PyObject *, PyObject *, PyObject *);

/* イテレータ関数 */
typedef PyObject *(*getiterfunc)(PyObject *);
typedef PyObject *(*iternextfunc)(PyObject *);

/* バッファプロトコルのインターフェース */
typedef struct {
    /* バッファプロトコル関連の定義 (必要に応じて追加) */
    int dummy;  /* プレースホルダ */
} PyBufferProcs;

/* 数値プロトコルの構造体 */
typedef struct {
    binaryfunc nb_add;
    binaryfunc nb_subtract;
    binaryfunc nb_multiply;
    binaryfunc nb_remainder;
    binaryfunc nb_divmod;
    ternaryfunc nb_power;
    unaryfunc nb_negative;
    unaryfunc nb_positive;
    unaryfunc nb_absolute;
    /* その他の数値メソッド (必要に応じて追加) */
} PyNumberMethods;

/* シーケンスプロトコルの構造体 */
typedef struct {
    lenfunc sq_length;
    binaryfunc sq_concat;
    PyObject *(*sq_item)(PyObject *, Py_ssize_t);
    /* その他のシーケンスメソッド (必要に応じて追加) */
} PySequenceMethods;

/* マッピングプロトコルの構造体 */
typedef struct {
    lenfunc mp_length;
    binaryfunc mp_subscript;
    int (*mp_ass_subscript)(PyObject *, PyObject *, PyObject *);
    /* その他のマッピングメソッド (必要に応じて追加) */
} PyMappingMethods;

/* Pythonの型オブジェクト定義 */
struct _typeobject {
    PyObject ob_base;                      /* PyObjectとして振る舞える */
    const char *tp_name;                   /* 型の名前 (モジュール名.クラス名) */
    Py_ssize_t tp_basicsize;               /* インスタンスの基本サイズ */
    Py_ssize_t tp_itemsize;                /* 可変長オブジェクトの追加アイテムサイズ */
    
    /* オブジェクトのメモリ管理 */
    void (*tp_dealloc)(PyObject *);        /* デストラクタ */
    void *tp_print;                        /* 廃止されたが互換性のために残す */
    
    /* オブジェクトの標準的な操作 */
    getattrfunc tp_getattr;                /* __getattr__ */
    setattrfunc tp_setattr;                /* __setattr__ */
    PyObject *(*tp_repr)(PyObject *);      /* __repr__ */
    
    /* メソッドスイート (型のvtable) */
    PyNumberMethods *tp_as_number;         /* 数値型メソッド */
    PySequenceMethods *tp_as_sequence;     /* シーケンス型メソッド */
    PyMappingMethods *tp_as_mapping;       /* マッピング型メソッド */
    
    /* その他の標準操作 */
    hashfunc tp_hash;                      /* __hash__ */
    ternaryfunc tp_call;                   /* __call__ */
    reprfunc tp_str;                       /* __str__ */
    getiterfunc tp_iter;                   /* __iter__ */
    iternextfunc tp_iternext;              /* __next__ */
    
    /* リッチ比較 */
    richcmpfunc tp_richcompare;            /* __eq__, __lt__ など */
    
    /* 属性管理 */
    PyObject *tp_dict;                     /* 属性辞書 */
    
    /* サブクラス化サポート */
    PyObject *tp_base;                     /* 基底クラス */
    PyObject *tp_bases;                    /* 基底クラスのタプル */
    
    /* 初期化、割り当て */
    PyObject *(*tp_new)(PyTypeObject *, PyObject *, PyObject *);  /* __new__ */
    int (*tp_init)(PyObject *, PyObject *, PyObject *);           /* __init__ */
    
    /* GCサポート */
    int (*tp_traverse)(PyObject *, int (*visit)(PyObject *, void *), void *);
    int (*tp_clear)(PyObject *);
};

/* 参照カウント操作 - 関数としてエクスポート */
void Py_INCREF(PyObject *op);
void Py_DECREF(PyObject *op);
void Py_XINCREF(PyObject *op);
void Py_XDECREF(PyObject *op);

/* 型チェックマクロ */
#define PyObject_TypeCheck(ob, tp) \
    (Py_TYPE(ob) == (tp) || PyType_IsSubtype(Py_TYPE(ob), (tp)))

/* 基本的なオブジェクト操作関数 */
PyObject* PyObject_Repr(PyObject *obj);
PyObject* PyObject_Str(PyObject *obj);
int PyObject_Compare(PyObject *o1, PyObject *o2);
PyObject* PyObject_RichCompare(PyObject *o1, PyObject *o2, int op);
int PyObject_RichCompareBool(PyObject *o1, PyObject *o2, int op);
Py_ssize_t PyObject_Hash(PyObject *obj);
int PyObject_IsTrue(PyObject *obj);
PyObject* PyObject_GetAttrString(PyObject *obj, const char *name);
int PyObject_SetAttrString(PyObject *obj, const char *name, PyObject *value);
int PyObject_HasAttrString(PyObject *obj, const char *name);

/* 型関連の関数 */
int PyType_IsSubtype(PyTypeObject *a, PyTypeObject *b);
void PyType_Ready(PyTypeObject *type);

/* None型の定義 */
extern PyObject _Py_NoneStruct;
extern PyObject *_Py_None;  /* LLVM IR向けのグローバルシンボル */
extern PyObject *Py_None;   /* LLVM IR用のエイリアス */
#define Py_None (&_Py_NoneStruct)

/* 真偽値の定義 */
extern PyObject _Py_TrueStruct;
extern PyObject _Py_FalseStruct;
extern PyObject *_Py_True;  /* LLVM IR向けのグローバルシンボル */
extern PyObject *_Py_False; /* LLVM IR向けのグローバルシンボル */
extern PyObject *Py_True;   /* LLVM IR用のエイリアス */
extern PyObject *Py_False;  /* LLVM IR用のエイリアス */
#define Py_True (&_Py_TrueStruct)
#define Py_False (&_Py_FalseStruct)

/* 比較演算子の定数 */
enum {
    Py_LT = 0,
    Py_LE = 1,
    Py_EQ = 2,
    Py_NE = 3,
    Py_GT = 4,
    Py_GE = 5
};

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_OBJECT_H */
