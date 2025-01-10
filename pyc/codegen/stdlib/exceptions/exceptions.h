#ifdef EXCEPTIONS_H
#define EXCEPTIONS_H

#include "../object.h"

// 例外を発生させる関数
void raise_exception(PyBaseExceptionObject* exc);

#endif // EXCEPTIONS_H
