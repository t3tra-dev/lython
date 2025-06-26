func.func @sum_list(%xs: memref<?xi32>) -> i32 {
  %c0 = arith.constant 0 : i32
  %sum = memref.alloca() : memref<i32>
  memref.store %c0, %sum[] : memref<i32>

  %n = memref.dim %xs, 0 : memref<?xi32>
  scf.for %i = %c0 to %n step %c1 {
    %x = memref.load %xs[%i] : memref<?xi32>
    %s_old = memref.load %sum[] : memref<i32>
    %s_new = arith.addi %s_old, %x : i32
    memref.store %s_new, %sum[] : memref<i32>
  }

  %ret = memref.load %sum[] : memref<i32>
  return %ret : i32
}
