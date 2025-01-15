#include "print.h"
#include "../stdtypes/str.h"
#include <stdio.h>

int print(const char* str) {
    if (str == NULL) {
        fprintf(stderr, "Error: NULL string\n");
        return -1;
    }
    return puts(str);
}

int print_object(PyObject* obj) {
    if (obj == NULL) {
        return print("None");
    }
    
    if (obj->ob_type != &PyString_Type) {
        fprintf(stderr, "TypeError: print() argument must be str\n");
        return -1;
    }
    
    PyStringObject* str_obj = (PyStringObject*)obj;
    return print(str_obj->value);
}
