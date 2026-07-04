> Please do not send pull requests to this repository.

# Lython - Python compiler toolchain based on LLVM

## セットアップ

### 必要なもの

- Python 3.12
- CMake 3.20+
- Ninja
- C++17 対応コンパイラ
- LLVM/MLIR 20 (`llvm@20` または `llvm-20-dev` / `libmlir-20-dev`)
- uv (Python パッケージマネージャ)

### ビルド手順

```bash
git clone https://github.com/t3tra-dev/lython.git
cd lython
uv sync

# macOS:
brew install llvm@20

# Ubuntu:
# wget https://apt.llvm.org/llvm.sh
# chmod +x llvm.sh
# sudo ./llvm.sh 20
# sudo apt-get install -y clang-20 lld-20 llvm-20-dev libmlir-20-dev mlir-20-tools

# Lython 本体
cmake -B build -S .
cmake --build build -j$(nproc)
```

### 実行

```bash
./build/bin/lyc jit examples/hello.py
./build/bin/lyc jit examples/fib.py
```

---

## ライセンス

本リポジトリのソースコードは、特記がない限り [MIT License](https://opensource.org/licenses/MIT) で配布されています。  
