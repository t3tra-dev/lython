// Python Dialect 基本型テスト
// !py.i32, !py.f64, !py.bool型の解析確認

module {
  // 基本型のテスト - 標準MLIR構文を使用
  func.func @test_python_types() {
    // Python基本型での変数宣言
    %int_val = arith.constant 42 : !py.i32
    %float_val = arith.constant 3.14 : !py.f64  
    %bool_val = arith.constant true : !py.bool
    
    // 型の表示テスト (戻り値なし)
    func.return
  }
}
