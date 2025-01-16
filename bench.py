import subprocess
import time
from typing import List, Tuple
from dataclasses import dataclass
from rich.console import Console
from rich.table import Table


@dataclass
class BenchmarkResult:
    name: str
    execution_time: float
    output: str


def run_command(command: str) -> Tuple[str, float]:
    start_time = time.time()
    result = subprocess.run(command.split(), capture_output=True, text=True)
    end_time = time.time()
    return result.stdout.strip(), end_time - start_time


def run_benchmarks() -> List[BenchmarkResult]:
    commands = {
        "Node.js": "node fib.js",
        "Bun": "bun fib.js",
        "Deno": "deno run fib.js",
        "C": "./cfib",
        "Python(pyc)": "./fib",
        "Python": "python fib.py",
        "Python(no GIL)": "python3.13t fib.py"
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

    sorted_results = sorted(results, key=lambda x: x.execution_time)
    fastest_time = sorted_results[0].execution_time

    for result in sorted_results:
        relative_speed = f"(x{result.execution_time / fastest_time:.2f})"
        time_str = f"{result.execution_time * 1000:.2f}ms {relative_speed}"
        table.add_row(result.name, time_str, result.output)

    console.print(table)


if __name__ == "__main__":
    results = run_benchmarks()
    display_results(results)
