py.func @sum_list(%xs: !py.list<i32>) -> !py.i32 {
  %s_init = py.constant 0 : !py.i32
  %s = py.var %s_init : !py.i32               // 可変変数
  %iter = py.iter %xs : !py.list<i32>         // イテレータ生成
  py.for %x in %iter : !py.i32 {
    %tmp = py.add %s, %x : !py.i32
    py.assign %s, %tmp : !py.i32
  }
  py.return %s : !py.i32
}
