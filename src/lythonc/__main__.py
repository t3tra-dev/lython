import os
import sys
from argparse import ArgumentParser

from lython import get_codegen, get_compiler

codegen = get_codegen()
compiler = get_compiler()


def main():
    parser = ArgumentParser()

    parser.add_argument(
        "-emit-llvm",
        help="Emit LLVM IR code",
        action='store_true'
    )
    parser.add_argument(
        "-o",
        help="Output file path",
        metavar="<output-path>",
        type=str
    )
    parser.add_argument(
        "input_file",
        help="Input Python file",
        metavar="<input-path>",
        type=str
    )

    args = parser.parse_args()

    if not args.emit_llvm and not args.o:
        print("Error: No option provided")
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' does not exist.")
        parser.print_help()
        sys.exit(1)

    if args.emit_llvm:
        print("Generate LLVM IR...")
        codegen.generate_llvm(args.input_file)
        print(f"Generated LLVM IR code at {args.input_file}.ll")

    elif args.o:
        output_file = args.o
        print(f"Compiling {args.input_file} to machine code...")
        compiler.ll2bin.compile(args.input_file, output_file)
        print(f"Compiled Python code to machine code at {output_file}")

    else:
        print("Error: Invalid option provided.")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
