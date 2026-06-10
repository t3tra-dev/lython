module {
  func.func private @LyEH_ThrowException(%header: memref<3xi64> {ly.ownership.object_header}, %message_header: memref<2xi64> {ly.ownership.object_header}, %message_bytes: memref<?xi8>) attributes {ly.ownership.transfer_args = [0]}

  func.func @LyException_New(%class_id: i64) -> memref<3xi64> attributes {ly.ownership.owned_results = [0]} {
    %one = arith.constant 1 : i64
    %layout_exception = arith.constant 5 : i64
    %refcount_slot = arith.constant 0 : index
    %layout_slot = arith.constant 1 : index
    %class_slot = arith.constant 2 : index

    %header = memref.alloc() {ly.ownership.object_header, ly.ownership.owned_local_object} : memref<3xi64>

    memref.store %one, %header[%refcount_slot] : memref<3xi64>
    memref.store %layout_exception, %header[%layout_slot] : memref<3xi64>
    memref.store %class_id, %header[%class_slot] : memref<3xi64>

    func.return %header : memref<3xi64>
  }

}
