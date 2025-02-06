import os
import sys
from argparse import ArgumentParser

from . import get_codegen, get_compiler

codegen = get_codegen()
compiler = get_compiler()


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "--emit-llvm",
        help="Emit LLVM IR code",
        metavar="<input-path>",
        type=str
    )
    parser.add_argument(
        "--compile",
        help="Compile Python code to machine code.",
        nargs=2,
        metavar=('<input-path>', '<output-path>'),
        type=str
    )
    parser.add_argument(
        "--dump-ast",
        help="Dump the AST of the Python code",
        metavar="<input-path>",
        type=str
    )

    args = parser.parse_args()

    if not args.emit_llvm and not args.compile and not args.dump_ast:
        print("Error: No option provided")
        parser.print_help()
        sys.exit(1)

    if args.emit_llvm:
        input_file = args.emit_llvm
    elif args.compile:
        input_file = args.compile[0]
    elif args.dump_ast:
        input_file = args.dump_ast

    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' does not exist.")
        parser.print_help()
        sys.exit(1)

    if args.emit_llvm:
        print("Generate LLVM IR...")
        codegen.generate_llvm(args.emit_llvm)
        print(f"Generated LLVM IR code at {args.emit_llvm}.ll")

    elif args.compile:
        input_file, output_file = args.compile
        print(f"Compiling {input_file} to machine code...")
        compiler.ll2bin.compile(input_file, output_file)
        print(f"Compiled Python code to machine code at {output_file}")

    elif args.dump_ast:
        print("The AST of the Python code is:")
        with open(args.dump_ast) as f:
            code = f.read()
        print(codegen.dump_ast(code))

    else:
        print("Error: Invalid option provided.")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
