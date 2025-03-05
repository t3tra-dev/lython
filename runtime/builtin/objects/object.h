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

/* デストラクタ関数 */
typedef void (*destructor)(PyObject *);

/* GCサポート用関数 */
typedef int (*visitproc)(PyObject *, void *);
typedef int (*traverseproc)(PyObject *, visitproc, void *);

/* インスタンス生成関数 */
typedef PyObject *(*newfunc)(PyTypeObject *, PyObject *, PyObject *);

/* 初期化関数 */
typedef int (*initproc)(PyObject *, PyObject *, PyObject *);

/* 属性ゲッター関数 (新スタイル) */
typedef PyObject *(*getattrofunc)(PyObject *, PyObject *);

/* 属性セッター関数 (新スタイル) */
typedef int (*setattrofunc)(PyObject *, PyObject *, PyObject *);

/* バッファプロトコル構造体 */
typedef struct {
    /* バッファプロトコル関連 */
    int dummy;
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
    destructor tp_dealloc;                 /* デストラクタ */
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
    getattrofunc tp_getattro;              /* __getattribute__ */
    setattrofunc tp_setattro;              /* __setattr__ */
    
    /* バッファプロトコル (廃止) */
    PyBufferProcs *tp_as_buffer;

    /* フラグ */
    long tp_flags;
    
    /* ドキュメント文字列 */
    const char *tp_doc;
    
    /* GCサポート */
    traverseproc tp_traverse;
    
    /* その他のメンバー */
    void *tp_clear;
    richcmpfunc tp_richcompare;            /* 比較関数 */
    
    /* 弱参照サポート */
    Py_ssize_t tp_weaklistoffset;
    
    /* イテレータサポート */
    getiterfunc tp_iter;
    iternextfunc tp_iternext;
    
    /* 属性アクセス */
    struct PyMethodDef *tp_methods;
    struct PyMemberDef *tp_members;
    struct PyGetSetDef *tp_getset;
    
    /* 継承サポート */
    PyObject *tp_base;
    PyObject *tp_dict;
    PyObject *tp_bases;
    
    /* 動的生成サポート */
    newfunc tp_new;
    initproc tp_init;
    
    /* その他のメンバー（必要に応じて） */
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
PyObject* PyObject_Call(PyObject *callable, PyObject *args, PyObject *kwargs);
void PyObject_ClearWeakRefs(PyObject *obj);

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

/* ガベージコレクション関連の型 */
typedef int (*visitproc)(PyObject *, void *);
typedef int (*traverseproc)(PyObject *, visitproc, void *);
typedef void (*destructor)(PyObject *);

/* NotImplemented 定数 */
extern PyObject _Py_NotImplementedStruct;
extern PyObject *Py_NotImplemented;
#define Py_NotImplemented (&_Py_NotImplementedStruct)

/* True/False 返却用の便利なマクロ */
#define Py_RETURN_TRUE return Py_NewRef(Py_True)
#define Py_RETURN_FALSE return Py_NewRef(Py_False)
#define Py_RETURN_NONE return Py_NewRef(Py_None)
#define Py_RETURN_NOTIMPLEMENTED return Py_NewRef(Py_NotImplemented)

/* メモリ確保関連マクロ */
#define Py_VISIT(o) do { if (o) { if (visit((PyObject *)(o), arg)) return 1; } } while(0)

/* 参照カウント関数 */
#define Py_NewRef(obj) _Py_NewRef((PyObject *)(obj))
#define Py_XNewRef(obj) _Py_XNewRef((PyObject *)(obj))

static inline PyObject* _Py_NewRef(PyObject *obj) {
    Py_INCREF(obj);
    return obj;
}

static inline PyObject* _Py_XNewRef(PyObject *obj) {
    Py_XINCREF(obj);
    return obj;
}

/* ゲッター・セッター関数型 */
typedef PyObject *(*getattrofunc)(PyObject *, PyObject *);
typedef int (*setattrofunc)(PyObject *, PyObject *, PyObject *);

/* 追加の関数型 */
typedef PyObject *(*getattrfunc)(PyObject *, char *);
typedef int (*setattrfunc)(PyObject *, char *, PyObject *);

/* デストラクタ関数 */
typedef void (*destructor)(PyObject *);

/* ゲッター関数 (非推奨) */
typedef PyObject *(*getattrfunc)(PyObject *, char *);

/* セッター関数 (非推奨) */
typedef int (*setattrfunc)(PyObject *, char *, PyObject *);

/* 属性ゲッター関数 (新スタイル) */
typedef PyObject *(*getattrofunc)(PyObject *, PyObject *);

/* 属性セッター関数 (新スタイル) */
typedef int (*setattrofunc)(PyObject *, PyObject *, PyObject *);

/* 表示関数 */
typedef PyObject *(*reprfunc)(PyObject *);

/* ハッシュ関数 */
typedef Py_hash_t (*hashfunc)(PyObject *);

/* 長さ取得関数 */
typedef Py_ssize_t (*lenfunc)(PyObject *);

/* 呼び出し関数 */
typedef PyObject *(*ternaryfunc)(PyObject *, PyObject *, PyObject *);

/* 二項演算子関数 */
typedef PyObject *(*binaryfunc)(PyObject *, PyObject *);

/* 単項演算子関数 */
typedef PyObject *(*unaryfunc)(PyObject *);

/* 比較関数 */
typedef PyObject *(*richcmpfunc)(PyObject *, PyObject *, int);

/* イテレータ関数 */
typedef PyObject *(*getiterfunc)(PyObject *);
typedef PyObject *(*iternextfunc)(PyObject *);

/* GCサポート用関数 */
typedef int (*visitproc)(PyObject *, void *);
typedef int (*traverseproc)(PyObject *, visitproc, void *);

/* インスタンス生成関数 */
typedef PyObject *(*newfunc)(PyTypeObject *, PyObject *, PyObject *);

/* 初期化関数 */
typedef int (*initproc)(PyObject *, PyObject *, PyObject *);

/* 便利なマクロ */
#define PyObject_HEAD_INIT(type) { 1, type }
#define Py_VISIT(o) do { if (o) { if (visit((PyObject *)(o), arg)) return 1; } } while(0)

#ifdef __cplusplus
}
#endif

#endif /* LYTHON_OBJECT_H */
