#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "functions.h"
#include "objects/object.h"
#include "objects/unicodeobject.h"
#include "objects/intobject.h"

/**
 * print関数 - オブジェクトを文字列として出力
 */
void print(PyObject *obj) {
    if (obj == NULL) {
        printf("None\n");
        return;
    }

    /* オブジェクトを文字列に変換 */
    PyObject *str_obj = PyObject_Str(obj);
    if (str_obj == NULL) {
        printf("<error converting to string>\n");
        return;
    }

    /* PyUnicodeObjectから文字列を取得 */
    if (PyUnicode_Check(str_obj)) {
        const char *str = PyUnicode_AsUTF8(str_obj);
        if (str) {
            printf("%s\n", str);
        } else {
            printf("<error accessing string data>\n");
        }
    } else {
        printf("<not a string object>\n");
    }

    /* 参照を解放 */
    Py_DECREF(str_obj);
}

/**
 * int2str関数 - 整数値を文字列オブジェクトに変換
 */
PyObject* int2str(int value) {
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%d", value);
    return PyUnicode_FromString(buffer);
}

/**
 * str2str関数 - 文字列オブジェクトをそのまま返す
 */
PyObject* str2str(PyObject *str) {
    if (!PyUnicode_Check(str)) {
        return PyObject_Str(str);
    }
    Py_INCREF(str);
    return str;
}

/**
 * create_string関数 - C文字列からPython文字列オブジェクトを生成
 */
PyObject* create_string(const char *str) {
    return PyUnicode_FromString(str);
}
