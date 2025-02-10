import subprocess
import time
from dataclasses import dataclass
from typing import List, Tuple

from rich.console import Console
from rich.table import Table


@dataclass
class BenchmarkResult:
    name: str
    execution_time: float
    output: str


def setup() -> None:
    # C言語の各最適化レベルでコンパイル
    optimization_levels = ["O0", "O1", "O2", "O3"]
    for opt in optimization_levels:
        subprocess.run(f"clang ./benchmark/cfib.c -o cfib_{opt} -{opt}".split())

    # LLVMの各最適化レベルでコンパイル
    for opt in optimization_levels:
        subprocess.run(f"clang ./benchmark/llfib.ll -o llfib_{opt} -{opt}".split())

    subprocess.run("python -m pyc --compile ./benchmark/pyfib.py pyfib".split())


def run_command(command: str) -> Tuple[str, float]:
    # ウォームアップ
    subprocess.run(command.split(), capture_output=True, text=True)

    # 複数回実行して平均を取る
    iterations = 5
    times = []

    for _ in range(iterations):
        start_time = time.perf_counter()
        result = subprocess.run(command.split(), capture_output=True, text=True)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    # 平均実行時間を計算
    avg_time = sum(times) / len(times)
    return result.stdout.strip(), avg_time


def run_benchmarks() -> List[BenchmarkResult]:
    commands = {
        "Node.js": "node ./benchmark/jsfib.js",
        "Bun": "bun ./benchmark/jsfib.js",
        "Deno": "deno run ./benchmark/jsfib.js",
        "C(O0)": "./cfib_O0",
        "C(O1)": "./cfib_O1",
        "C(O2)": "./cfib_O2",
        "C(O3)": "./cfib_O3",
        "LLVM(O0)": "./llfib_O0",
        "LLVM(O1)": "./llfib_O1",
        "LLVM(O2)": "./llfib_O2",
        "LLVM(O3)": "./llfib_O3",
        "Python(pyc)": "./pyfib",
        "Python": "python ./benchmark/pyfib.py",
        "Python(no GIL)": "python3.13t -X gil=1 ./benchmark/pyfib.py"
    }

    results = []
    for name, cmd in commands.items():
        output, execution_time = run_command(cmd)
        results.append(BenchmarkResult(name, execution_time, output))

    return results


def display_results(results: List[BenchmarkResult]):
    console = Console()
    table = Table(title="🚀 Benchmark Results")

    table.add_column("runtime", style="cyan")
    table.add_column("time", style="green")
    table.add_column("result", style="yellow")

    pyc_time = next((result.execution_time for result in results if result.name == "Python(pyc)"), None)

    sorted_results = sorted(results, key=lambda x: x.execution_time)

    for result in sorted_results:
        # "Python(pyc)"を基準に相対速度を計算
        relative_speed = f"(x{result.execution_time / pyc_time:.2f})"  # type: ignore
        time_str = f"{result.execution_time * 1000:.2f}ms {relative_speed}"
        table.add_row(result.name, time_str, f"{result.output} ")

    console.print(table)


if __name__ == "__main__":
    setup()
    results = run_benchmarks()
    display_results(results)
