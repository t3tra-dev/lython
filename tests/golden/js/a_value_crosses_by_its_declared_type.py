# What: values cross to and from the JavaScript host by the types the `js`
# stub declares at each end -- numbers, ints (a BigInt past 2**53), bools,
# strs of every storage width, a lone surrogate -- and a host exception
# arrives as RuntimeError with its JavaScript name and message.
#
# Why run: only the host shows what crossed; the compiler sees types.
from js import JSON, Math, Number, console

print(Math.sqrt(2.0), Math.PI, Math.max(1.0, 5.5, 3.0))
print(Math.floor(7.9) + 1)
print(JSON.stringify("é日🐍"), JSON.stringify(True), JSON.stringify(41 + 1))
escaped: str = JSON.stringify(chr(0xD800) + "x")
print(len(escaped), repr(escaped))
print("日🐍" in JSON.stringify("é日🐍"))
# A Number while a double holds the int exactly, a BigInt past that:
# isInteger is false for a BigInt.
print(Number.isInteger(2**53 - 1), Number.isInteger(2**53), Number.isInteger(2**70))
try:
    JSON.stringify(2**70)
except RuntimeError as error:
    print("BigInt:", error)
try:
    JSON.parse("{")
except RuntimeError as error:
    print("caught:", error)
# Last, because node's console writes a pipe asynchronously on macOS and a
# print after it could overtake it. The BigInts' digits, as the host has them.
console.log(2**53 + 1, 2**70, -(2**64), 2**53 - 1)
