CC = clang
CFLAGS = -Wall -Wextra -I runtime/builtin
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
	rm -f runtime.o
	rm -f $(RUNTIME_DIR)/*.o

.PHONY: all clean
