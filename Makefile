CC = clang
CFLAGS = -Wall -Wextra -O2 -I runtime/builtin -I runtime/builtin/objects $(shell pkg-config --cflags bdw-gc)
LDFLAGS = $(shell pkg-config --libs bdw-gc)
RUNTIME_DIR = runtime/builtin
OBJECTS_DIR = $(RUNTIME_DIR)/objects

# 自動的にすべてのソースファイルを検出
OBJECTS_SOURCES := $(wildcard $(OBJECTS_DIR)/*.c)
OBJECTS_OBJECTS := $(OBJECTS_SOURCES:.c=.o)

# functions.cも含める
RUNTIME_SOURCES := $(wildcard $(RUNTIME_DIR)/*.c)
RUNTIME_OBJECTS := $(RUNTIME_SOURCES:.c=.o)

# 全てのオブジェクトファイル
ALL_OBJECTS := $(OBJECTS_OBJECTS) $(RUNTIME_OBJECTS)

# ビルドターゲット
all: runtime.o

# ランタイムライブラリのビルド
runtime.o: $(ALL_OBJECTS)
	ld -r $^ -o $@

# objectsディレクトリのオブジェクトファイルをビルド
$(OBJECTS_DIR)/%.o: $(OBJECTS_DIR)/%.c $(OBJECTS_DIR)/%.h
	$(CC) $(CFLAGS) -c $< -o $@

# runtimeディレクトリのオブジェクトファイルをビルド
$(RUNTIME_DIR)/%.o: $(RUNTIME_DIR)/%.c $(RUNTIME_DIR)/%.h
	$(CC) $(CFLAGS) -c $< -o $@

# クリーンアップ
clean:
	find . -name '__pycache__' -type d -exec rm -r {} +
	rm -f runtime.o
	rm -f $(OBJECTS_DIR)/*.o
	rm -f $(RUNTIME_DIR)/*.o

.PHONY: all clean
