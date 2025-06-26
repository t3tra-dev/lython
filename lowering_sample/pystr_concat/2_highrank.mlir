func.func @greet(%name: !rt.pyobj) -> !rt.pyobj {
  %cst = rt.py_const_str "Hello, "               // 文字列リテラルをランタイム定数に
  %0   = rt.py_call @PyStr_Concat(%cst, %name)   // CPython 互換連結 API
         : (!rt.pyobj, !rt.pyobj) -> !rt.pyobj
  return %0 : !rt.pyobj
}
