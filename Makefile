CC = clang
CFLAGS = -Wall -Wextra -O2 -I runtime/builtin $(shell pkg-config --cflags bdw-gc)
LDFLAGS = $(shell pkg-config --libs bdw-gc)
RUNTIME_DIR = runtime/builtin

# ビルドターゲット
all: runtime.o

# ランタイムライブラリのビルド
runtime.o: $(RUNTIME_DIR)/functions.o $(RUNTIME_DIR)/types.o
	ld -r $^ -o $@

# 個別のオブジェクトファイル
$(RUNTIME_DIR)/functions.o: $(RUNTIME_DIR)/functions.c $(RUNTIME_DIR)/functions.h $(RUNTIME_DIR)/types.h
	$(CC) $(CFLAGS) -c $< -o $@

$(RUNTIME_DIR)/types.o: $(RUNTIME_DIR)/types.c $(RUNTIME_DIR)/types.h
	$(CC) $(CFLAGS) -c $< -o $@

# クリーンアップ
clean:
	find . -name '__pycache__' -type d -exec rm -r {} +
	rm -f runtime.o
	rm -f $(RUNTIME_DIR)/*.o

.PHONY: all clean
