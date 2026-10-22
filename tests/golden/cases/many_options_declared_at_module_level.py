# WHAT: a script that declares thirty-two options at module level and
# parses a command line.
#
# WHY THIS IS RUN AND NOT CHECKED AT A LOWER LAYER: the defects were a
# refusal of the whole program and its compile time; running it shows both
# are gone and the parse is right.
#
# ⛔ Each `parser.add_argument(...)` copied argparse's `_add_argument` (450
# nodes) into the module body -- 770 blocks a call -- because the method was
# called by symbol only when every argument was exactly its parameter's
# type, and the wrapper hands it the caller's narrower ones. Sixteen calls
# compiled; thirty-two exceeded the ownership check's 20000-state budget, and
# took 125 s where this takes about 13.
import argparse

parser = argparse.ArgumentParser(prog="many")
parser.add_argument('--o0', type=int, default=0)
parser.add_argument('--o1', type=int, default=1)
parser.add_argument('--o2', type=int, default=2)
parser.add_argument('--o3', type=int, default=3)
parser.add_argument('--o4', type=int, default=4)
parser.add_argument('--o5', type=int, default=5)
parser.add_argument('--o6', type=int, default=6)
parser.add_argument('--o7', type=int, default=7)
parser.add_argument('--o8', type=int, default=8)
parser.add_argument('--o9', type=int, default=9)
parser.add_argument('--o10', type=int, default=10)
parser.add_argument('--o11', type=int, default=11)
parser.add_argument('--o12', type=int, default=12)
parser.add_argument('--o13', type=int, default=13)
parser.add_argument('--o14', type=int, default=14)
parser.add_argument('--o15', type=int, default=15)
parser.add_argument('--o16', type=int, default=16)
parser.add_argument('--o17', type=int, default=17)
parser.add_argument('--o18', type=int, default=18)
parser.add_argument('--o19', type=int, default=19)
parser.add_argument('--o20', type=int, default=20)
parser.add_argument('--o21', type=int, default=21)
parser.add_argument('--o22', type=int, default=22)
parser.add_argument('--o23', type=int, default=23)
parser.add_argument('--o24', type=int, default=24)
parser.add_argument('--o25', type=int, default=25)
parser.add_argument('--o26', type=int, default=26)
parser.add_argument('--o27', type=int, default=27)
parser.add_argument('--o28', type=int, default=28)
parser.add_argument('--o29', type=int, default=29)
parser.add_argument('--o30', type=int, default=30)
parser.add_argument('--o31', type=int, default=31)
args = parser.parse_args(['--o3', '30', '--o31', '310'])
print(repr(args.o0), repr(args.o3), repr(args.o31))
