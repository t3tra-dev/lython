`runtime/_bltinmodule.ll`
- 8958行目~9150行目 `print()` の実装
`runtime/_bltinmodule.ll`
- 8958行目~9150行目 `print()` の実装

bench
```bash
# build runtime
make clean && make

# C
opt_levels=("O0" "O1" "O2" "O3")
for opt in "${opt_levels[@]}"; do
    clang ./benchmark/cfib.c -o cfib_"$opt" -"$opt"
done

# LLVM
for opt in "${opt_levels[@]}"; do
    clang ./benchmark/llfib.ll -o llfib_"$opt" -"$opt"
done

# Lython
source .venv/bin/activate && python -m lythonc ./benchmark/pyfib.py -o pyfib

# run bench
hyperfine -w 30 -r 5 --sort command './pyfib' './cfib_O0' './cfib_O1' './cfib_O2' './cfib_O3' './llfib_O0' './llfib_O1' './llfib_O2' './llfib_O3'
```
