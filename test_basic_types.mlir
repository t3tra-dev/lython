// 基本型のテスト - Python Dialect
// !py.i32, !py.f64, !py.bool型の動作確認

module {
  // 基本型のテスト関数 (標準MLIR構文)
  func.func @test_types() {
    // 各型の変数宣言テスト
    %int_val = arith.constant 42 : !py.i32
    %float_val = arith.constant 3.14 : !py.f64
    %bool_val = arith.constant true : !py.bool
    
    func.return
  }
}
