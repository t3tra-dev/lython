// Python Dialect basic test
module {
  py.func @test_add() -> !py.i32 {
    %c1 = py.constant 1 : !py.i32
    %c2 = py.constant 2 : !py.i32
    %result = py.add %c1, %c2 : !py.i32
    py.return %result : !py.i32
  }
}