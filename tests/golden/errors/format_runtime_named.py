# 実行時テンプレートの名前フィールドは静的に束縛できないため実行時に拒否する。
# (CPython は KeyError; Lython は auto-only 制限を ValueError で通知する)
def tpl() -> str:
    return "{name}"

print(tpl().format("v"))
