py.func @greet(%name: !py.obj<str>) -> !py.obj<str> {
  %cst = py.constant "Hello, " : !py.obj<str>
  %0 = py.add %cst, %name : (!py.obj<str>, !py.obj<str>) -> !py.obj<str>
  py.return %0 : !py.obj<str>
}
