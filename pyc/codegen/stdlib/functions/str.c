#include "../object.h"
#include "../stdtypes/str.h"

PyObject* str(PyObject* obj) {
    if (obj == NULL) {
        return PyString_FromString("None");
    }
    
    // オブジェクトのメソッドテーブルから__str__を取得
    PyMethodTable* methods = obj->methods;
    if (methods && methods->str) {
        PyObject* result = methods->str(obj);
        if (result == NULL) {
            return PyString_FromString("<error>");
        }
        return result;
    }
    
    // __str__が実装されていない場合はデフォルトの文字列表現を返す
    return default_str(obj);
}