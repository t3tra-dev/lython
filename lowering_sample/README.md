### 補足 - 実装時の要点

| フェーズ            | 主な注意点                                                             |
| ------------------- | ---------------------------------------------------------------------- |
| **Python Dialect**  | *型付き*AST に基づき必ず型が確定；ネイティブか PyObject かを属性で判別 |
| **SCF/Tensor 生成** | `py.for` を `scf.for` へ、リストは `memref<?xi32>` へ _bufferize_      |
| **Memref/Arith**    | アドレス計算を `memref.load/store` → `llvm.getelementptr` に自然変換   |
| **LLVM Dialect**    | `llvm.func` 生成時に `gc "shadow-stack"` を付与すれば GC 統合も可      |
| **LLVM IR**         | `opt -O2` 相当を掛ければループは vectorize / unroll など最適化可       |

これらの断片を基準に、実際の実装では TableGen → C++/Python Pass → MLIR Tools の順でパスを積み上げていくことになります。
