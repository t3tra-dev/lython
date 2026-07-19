# 実行時 spec の不正 type code は CPython と同じ ValueError で落ちる。
def spec_for(flag: bool) -> str:
    if flag:
        return "d"
    return ".2f"

x = 3.14
print(f"{x:{spec_for(False)}}")
print(f"{x:{spec_for(True)}}")
