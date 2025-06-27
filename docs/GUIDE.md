以下では、先に提示したトップレベル構造を踏まえ、**各コンポーネントを実装する際の技術的ディテール**をできる限り具体的に—章単位で整理します。コード断片はあくまで**雛形**ですが、実装着手時に“写経”できる粒度を意識しました。

---

## 1 📦 ビルド & リポジトリ基盤

```text
lython/
├─ CMakeLists.txt          # ルート。LLVM/MLIR の外部ビルドを in-tree import
├─ cmake/
│   └─ LLVMExternalProject.cmake  # FetchContent or ExternalProject_Add
└─ third_party/
    └─ llvm-project/       # version tag を pin。浅い clone 推奨
```

```cmake
# CMakeLists.txt (抜粋)
cmake_minimum_required(VERSION 3.24)
project(Lython LANGUAGES C CXX)

set(LLVM_ENABLE_PROJECTS "mlir;llvm")
set(LLVM_EXTERNAL_PROJECTS ${LLVM_ENABLE_PROJECTS})

include(cmake/LLVMExternalProject.cmake)     # ← llvm-project をビルド
add_subdirectory(src)                        # フロント/ランタイム
```

- **MLIR Python バインディング**: `-DLLVM_ENABLE_BINDINGS=ON -DMLIR_INCLUDE_IN_BINDINGS=ON` を指定。Python wheel 化は `scikit-build-core` or `pybind11`。

---

## 2 📝 Frontend ─ Parser & TypedAST

### 2-1 TypedAST ノード定義

```python
# src/lython/frontend/ast_nodes.py
from __future__ import annotations
import ast
from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class Ty:
    """型のメタ情報 (モノタイプ/ユニオン/ジェネリックの基底)"""
    py_name: str          # 'int', 'str', 'list[int]' など
    llvm: Optional[str]   # 'i32', '%pyobj*' など

@dataclass
class TypedNode:
    """全 TypedAST ノードの共通ベース"""
    node: ast.AST         # 元 CPython AST ノード
    ty: Ty                # 推論結果
    span: tuple[int, int] # (lineno, col)
```

サブクラス例:

```python
@dataclass
class TypedBinOp(TypedNode):
    op: ast.operator
    lhs: TypedNode
    rhs: TypedNode
```

### 2-2 型推論アルゴリズム

1. **収集フェーズ**

   - `TypeCollector` が関数毎に **シンボル表**を作成
   - `@native` / 型注釈 / リテラル から「既知型」を集め _制約_ として保存

2. **解決フェーズ (solver)**

   - Hindley–Milner 風 _unification_（ただしジェネリック深追いは避ける）
   - バリアント: 「不一致ならフォールバックして PyObject 型へ昇格」

3. **チェックフェーズ**

   - 未解決の型変数が残れば `Any` -> 互換モード、`--strict` ならエラー
   - `native` 領域では必ず具象型が必要

```python
# frontend/typing/solver.py (概略)
class ConstraintSolver:
    def unify(self, a: Ty, b: Ty):
        match (a, b):
            case (Ty(py_name='int'), Ty(py_name='int')): return a
            case (Ty(py_name='int'), Ty(py_name='float')):
                return Ty('float', 'double')  # 拡張
            case _ if 'Any' in (a.py_name, b.py_name):
                return a if a.py_name != 'Any' else b
            case _:
                raise TypeError(f"型不整合: {a.py_name} vs {b.py_name}")
```

### 2-3 エラー収集

```python
class ErrorReporter:
    def __init__(self): self.errs: list[str] = []
    def add(self, msg: str, span): self.errs.append(f"{span[0]}:{span[1]} {msg}")

# 訪問中
if isinstance(lhs.ty, IntTy) and isinstance(rhs.ty, StrTy):
    reporter.add("int と str の加算は未定義です", node.span)
```

---

## 3 🔧 native API

### 3-1 公開 API

```python
# src/lython/native/__init__.py
from typing import TypeVar, Generic
T = TypeVar("T")

class i32: pass
class f64: pass
# ... (スタブ)

def to_native(value, ty): ...
def to_pytype(value, py_cls): ...

def native(func=None, /, *, inline=False):
    def decorator(f):
        f.__ly_native__ = True
        f.__ly_inline__ = inline
        return f
    return decorator(func) if func else decorator
```

### 3-2 Frontend での検出

```python
# visit_FunctionDef
if getattr(node, "__ly_native__", False):
    symtab.add_native_func(node.name, sig)
    return PythonBuilder.emit_native_def(node, sig, body_ir)
```

---

## 4 🏗 MLIR Dialects

### 4-1 ODS (TableGen) での Python Dialect 定義

```tablegen
// dialects/python/PythonOps.td
def Python_Dialect : Dialect {
  let name = "py";
  let cppNamespace = "py";
}

def Py_AddOp : Op<"add", [SameOperandsAndResultType]> {
  let summary = "Python add (int/str/list ...)";
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results   = (outs AnyType:$res);
  let builders  = [OpBuilder<(ins "Value":$lhs, "Value":$rhs)>];
}
```

### 4-2 Dialect 実装 (C++)

```cpp
// PYDialect.cpp
#include "Python/PythonDialect.h"
#include "Python/PythonOpsDialect.h"
using namespace mlir;
void py::PythonDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "Python/PythonOps.cpp.inc"
        >();
}
```

MLIR パス挿入例:

```cpp
// lowering/LowerAdd.cpp
struct LowerAdd : OpRewritePattern<py::AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(py::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto lhs = op.getLhs();  // Value
    auto rhs = op.getRhs();
    // 例: int の場合は arith.addi
    if (lhs.getType().isa<IntegerType>()) {
        rewriter.replaceOpWithNewOp<arith::AddIOp>(op, lhs, rhs);
        return success();
    }
    return failure();
  }
};
```

---

## 5 🧩 IR Builder 抽象層

```python
# src/lython/ir/builder.py
from abc import ABC, abstractmethod
class IRBuilder(ABC):
    @abstractmethod
    def add(self, a, b): ...
    @abstractmethod
    def ret(self, v): ...

class MLIRBuilder(IRBuilder):
    def __init__(self, ctx, block):
        self.ctl = ctx
        self.block = block
        self.rewriter = ir.InsertionPoint(block)

    def add(self, a, b):
        return arith.AddIOp(a.type, a, b).result

class LLVMTextBuilder(IRBuilder):
    def add(self, a, b):
        tmp = self.new_tmp('i32')
        self.emit(f"{tmp} = add i32 {a}, {b}")
        return tmp
```

利用例 (Visitor 内):

```python
value = self.builder.add(lhs_val, rhs_val)
self.builder.ret(value)
```

---

## 6 🚚 Pass Library

### 6-1 登録

```c++
// passes/Passes.cpp
#include "mlir/Pass/Pass.h"
std::unique_ptr<mlir::Pass> createLowerPyToSCF();
static PassRegistration<LowerPyToSCF> reg("lower-py-scf",
    "Lower Python Dialect to scf/tensor");
```

### 6-2 分割パス順

```
py.func -> (py.add / py.call / ...)        # Python Dialect
     ↓  LowerPyToSCF
scf.for / tensor.insert / linalg.generic   # 高位 MLIR
     ↓  Bufferize / Canonicalize
memref.load/store + arith/affine           # 低位 MLIR
     ↓  ConvertToLLVMDialect
llvm.*
```

---

## 7 🛠 Backend & CodeGen

### 7-1 CLI コマンド

```python
# cli/lythonc
parser.add_argument("-emit-llvm", action="store_true")
parser.add_argument("-O", type=int, default=2)
args = parser.parse_args()

module = compile_to_mlir(input_file)
llvm_mod = lower_to_llvm(module, opt_level=args.O)
if args.emit_llvm:
    print(llvm_mod)                      # .ll を stdout
else:
    write_object(llvm_mod, args.output) # clang-like driver
```

### 7-2 JIT (optional)

```python
from mlir.execution_engine import ExecutionEngine
ee = ExecutionEngine(llvm_mod, opt_level=3)
fn = ee.invoke("main", arg0, arg1)
```

---

## 8 ⚙ Runtime & GC

### 8-1 PyObject ヘッダ

```c
typedef struct _PyObject {
    uint32_t ob_refcnt;
    struct _PyTypeObject *ob_type;
} PyObject;
```

### 8-2 Shadow-Stack ルート登録

```llvm
define void @foo() gc "shadow-stack" {
entry:
  %root = alloca i8*
  call void @llvm.gcroot(i8** %root, i8* null)
  ; 生成した PyObject* を store
  store i8* %obj, i8** %root
  ...
}
```

### 8-3 参照カウント互換

```c
static inline void Py_INCREF(PyObject *o) { ++o->ob_refcnt; }
static inline void Py_DECREF(PyObject *o) {
    if (--o->ob_refcnt == 0) py_dealloc(o);
}
```

GC Migration Plan:

| フェーズ | 変更点                             |
| -------- | ---------------------------------- |
| **P1**   | `malloc`+参照カウント (循環無視)   |
| **P2**   | `gc "shadow-stack"` + precise mark |
| **P3**   | Statepoint + parallel GC           |

---

## 9 🎯 エンドツーエンド最小サンプル

```python
# sample.py
from native import native, i32

@native
def add(a: i32, b: i32) -> i32:
    return a + b

def main():
    x: int = 40
    y: int = 2
    print(add(x, y))

if __name__ == "__main__":
    main()
```

```shell
$ lythonc sample.py -emit-llvm -O2 -o sample.ll
$ clang sample.ll runtime/liblython_runtime.a -o sample
$ ./sample
42
```

---

以上が **Lython** 各コンポーネントの詳細設計ガイドです。
