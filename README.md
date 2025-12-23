> Please do not send pull requests to this repository.

# Lython - Python compiler toolchain based on LLVM

> [!TIP]
> Searching for **pyc**? You are in the right repo. **pyc** has been renamed to **Lython**.

現在、Lython を完全にリライトする作業を行っています。旧実装は `legacy` ブランチを参照してください。

---

## セットアップ

### 必要なもの

- Python 3.12
- CMake 3.20+
- Ninja
- C++17 対応コンパイラ
- uv (Python パッケージマネージャ)
- nanobind, pybind11

### ビルド手順

```bash
git clone --recurse-submodules https://github.com/t3tra-dev/lython.git
cd lython
uv sync

# LLVM/MLIR (初回のみ、時間がかかります)
uv run cmake -B third_party/llvm-project/build -S third_party
uv run cmake --build third_party/llvm-project/build

# Lython 本体
uv run cmake -B build -S .
uv run cmake --build build
```

### 実行

```bash
./build/bin/lyc jit examples/hello.py
./build/bin/lyc jit examples/fib.py
```

---

## ライセンス

本リポジトリのソースコードは、特記がない限り [MIT License](https://opensource.org/licenses/MIT) で配布されています。  
詳細はソースコード内の記述 (`lython/__init__.py` など) をご参照ください。
