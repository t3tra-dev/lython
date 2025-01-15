#ifndef STRING_H
#define STRING_H

#include "../object.h"

// 文字列用のメソッドテーブル
extern PyMethodTable string_method_table;

// PyString_Typeの宣言
extern TypeObject PyString_Type;

// PyString_FromString関数の宣言
PyObject* PyString_FromString(const char* str);

// 文字列オブジェクト用の__str__メソッド
PyObject* string_str(PyObject* self);

#endif // STRING_H
