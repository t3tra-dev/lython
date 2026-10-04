"""ctest check: the native runtime lyc embeds, precompiled at build time, is
the runtime it would have lowered itself.

Compiles each program twice with one lyc -- as shipped, and under
LYTHON_ABLATE_PRECOMPILED_RUNTIME=1, which lowers the runtime in the compile --
and requires the two LLVM modules to be the same module.

"The same" up to what the two links cannot agree on and nothing reads:
  - the numbers in private and internal names (`@assert_msg_7`), which a
    counter in the lowering hands out in process order -- a fresh build-time
    process and a compile that lowered the program first count differently.
    Such a global is named by its definition instead;
  - the `.N` suffix the linker gives a colliding name, likewise;
  - the order of top-level definitions and of `; preds =` comments.
Each function body is otherwise compared instruction for instruction.

It also requires that the shipped run USED the precompiled runtime and the
ablated one did not, from the perf trace: a lyc whose embedded runtime stopped
matching its own target would fall back silently and pass this comparison by
comparing a path with itself.

Exit 0 = equivalent, 1 = not, 2 = could not run.
"""

import collections
import hashlib
import os
import pathlib
import re
import subprocess
import sys

SUFFIXED = re.compile(r"(@[\w$]+)\.[0-9]+\b")


def normalize(text: str) -> str:
    names = {}
    for match in re.finditer(r"^@([\w.$]+) = (?:private|internal) (.*)$", text,
                             re.M):
        names[match.group(1)] = "G_" + hashlib.sha1(
            match.group(2).encode()).hexdigest()[:16]
    text = re.sub(r"@([\w.$]+)",
                  lambda m: "@" + names.get(m.group(1), m.group(1)), text)
    text = re.sub(r"\s*; preds = .*$", "", text, flags=re.M)
    suffixed = {}
    for match in re.finditer(
            r"^(define|declare) ([^@]*)@([\w$]+)\.([0-9]+)\(", text, re.M):
        start = match.start()
        end = (text.find("\n}\n", start) if match.group(1) == "define"
               else text.find("\n", start))
        body = SUFFIXED.sub(r"\1", text[start:end])
        suffixed[f"{match.group(3)}.{match.group(4)}"] = (
            match.group(3) + "#" + hashlib.sha1(body.encode()).hexdigest()[:12])
    return re.sub(r"@([\w$]+\.[0-9]+)\b",
                  lambda m: "@" + suffixed.get(m.group(1), m.group(1)), text)


def entities(text: str) -> "collections.Counter[str]":
    out: "collections.Counter[str]" = collections.Counter()
    current = None
    for line in text.splitlines():
        if current is not None:
            current.append(line)
            if line == "}":
                out["\n".join(current)] += 1
                current = None
            continue
        if line.startswith("define "):
            current = [line]
            continue
        if line.strip() and not line.startswith(";"):
            out[line] += 1
    return out


def compile_ir(lyc: pathlib.Path, source: pathlib.Path,
               ablate: bool) -> "tuple[str, str] | None":
    env = dict(os.environ, LYTHON_PERF="1")
    if ablate:
        env["LYTHON_ABLATE_PRECOMPILED_RUNTIME"] = "1"
    else:
        env.pop("LYTHON_ABLATE_PRECOMPILED_RUNTIME", None)
    result = subprocess.run([str(lyc), str(source), "--emit-llvm", "-o", "-"],
                            capture_output=True, text=True, env=env,
                            stdin=subprocess.DEVNULL, timeout=600)
    if result.returncode != 0:
        print(f"{source.name}: lyc failed:\n{result.stderr[-2000:]}",
              file=sys.stderr)
        return None
    return result.stdout, result.stderr


def main() -> int:
    lyc = pathlib.Path(sys.argv[1]).resolve()
    sources = [pathlib.Path(arg).resolve() for arg in sys.argv[2:]]
    status = 0
    for source in sources:
        shipped = compile_ir(lyc, source, ablate=False)
        lowered = compile_ir(lyc, source, ablate=True)
        if shipped is None or lowered is None:
            return 2
        if "phase=link-runtime.precompiled " not in shipped[1]:
            print(f"{source.name}: the shipped compile did not use the "
                  "precompiled runtime, so there is nothing to compare. Is the "
                  "embedded one built for another target?", file=sys.stderr)
            status = 1
            continue
        if "phase=link-runtime.lowered " not in lowered[1]:
            print(f"{source.name}: the ablated compile did not lower the "
                  "runtime itself", file=sys.stderr)
            status = 1
            continue
        a = entities(normalize(shipped[0]))
        b = entities(normalize(lowered[0]))
        if a != b:
            print(f"{source.name}: the precompiled runtime differs from the "
                  f"one lowered in the compile ({sum((a - b).values())} "
                  f"entities only in the first, {sum((b - a).values())} only "
                  f"in the second); first of them:", file=sys.stderr)
            for entity in list((a - b))[:2] + list((b - a))[:2]:
                print(entity[:400], file=sys.stderr)
            status = 1
        else:
            print(f"{source.name}: equivalent")
    return status


if __name__ == "__main__":
    sys.exit(main())
