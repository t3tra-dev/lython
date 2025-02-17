import subprocess

from .. import codegen


def compile(input_path: str, output_path: str):
    codegen.generate_llvm(input_path)

    libs = subprocess.check_output(
        "pkg-config --libs --cflags bdw-gc",
        shell=True,
        text=True
    ).strip().split()

    subprocess.run([
        "clang",
        "-O2",
        f"{input_path}.ll",
        "runtime.o",
        "-o", output_path,
        *libs
    ], check=True)
