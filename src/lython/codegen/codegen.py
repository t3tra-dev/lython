import ast

from .generator import IRGenerator


def generate_llvm(source_file: str) -> None:
    """Pythonソースコードからのコード生成"""
    with open(source_file) as f:
        source = f.read()

    # ASTの生成
    tree = ast.parse(source)

    # LLVM IRの生成
    generator = IRGenerator()
    ir_code = generator.generate(tree)

    # 出力ファイルに書き込み
    with open(f"{source_file}.ll", "w") as f:
        f.write(ir_code)
