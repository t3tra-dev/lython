> Please do not send pull requests to this repository.

# Lython - Python compiler toolchain based on LLVM

## セットアップ

### 必要なもの

- Python 3.12
- CMake 3.20+
- Ninja
- C++17 対応コンパイラ
- LLVM/MLIR 22 (`llvm`/`llvm@22` または `llvm-22-dev` / `libmlir-22-dev`)
- uv (Python パッケージマネージャ)

### ビルド手順

```bash
git clone https://github.com/t3tra-dev/lython.git
cd lython
uv sync

# macOS:
brew install llvm

# Ubuntu:
# wget https://apt.llvm.org/llvm.sh
# chmod +x llvm.sh
# sudo ./llvm.sh 22
# sudo apt-get install -y clang-22 lld-22 llvm-22-dev libmlir-22-dev mlir-22-tools

# Lython 本体
cmake -B build -S .
cmake --build build -j$(nproc)
```

### 実行

```bash
./build/bin/lyc jit examples/hello.py
./build/bin/lyc jit examples/fib.py
```

### テスト

`build/` は既定で ctest スイート (`LYTHON_BUILD_TESTS=ON`) を含む。GoogleTest
ユニット、golden テスト、examples の smoke テストが走る。

```bash
ctest --test-dir build -j$(nproc) --output-on-failure
```

### Fuzzing

libFuzzer ベースの fuzzer (`fuzz_parser` / `fuzz_emitter` / `fuzz_pipeline`)
はツリー全体を ASan + カバレッジで計装するため、通常ビルドとは別ディレクトリを
使う。upstream clang が必要 (macOS では Homebrew LLVM の clang を CMake が
自動選択する。AppleClang は不可)。

```bash
# 専用ビルドディレクトリを構成してビルド
# (Ubuntu では -DCMAKE_C_COMPILER=clang-22 -DCMAKE_CXX_COMPILER=clang++-22 を追加)
cmake -B build-fuzz -S . \
  -DLYTHON_ENABLE_FUZZERS=ON -DLYTHON_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build-fuzz -j$(nproc) \
  --target fuzz_parser fuzz_emitter fuzz_pipeline

# チェックイン済みコーパスの回帰チェック (無変異リプレイ)
ctest --test-dir build-fuzz -j$(nproc) --output-on-failure

# 探索実行 (未計装の system dylib との併用で ASAN_OPTIONS が必要)
ASAN_OPTIONS=detect_leaks=0:detect_container_overflow=0:allow_user_poisoning=0 \
  ./build-fuzz/bin/fuzz_pipeline -max_total_time=300 -timeout=10 -rss_limit_mb=4096 \
    -dict=fuzz/dictionaries/python.dict \
    fuzz/corpora/fuzz_pipeline tests/golden/cases
```

見つかった crash 入力は診断化して修正し、`fuzz/corpora/<harness>/` に回帰用
シードとして追加する。

---

## ライセンス

本リポジトリのソースコードは、特記がない限り [MIT License](https://opensource.org/licenses/MIT) で配布されています。  
