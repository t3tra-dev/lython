import unicodedata

print(unicodedata.category("A"))
print(unicodedata.category("a"))
print(unicodedata.category("あ"))
print(unicodedata.category("漢"))
print(unicodedata.category(" "))
print(unicodedata.category("😀"))
print(unicodedata.category("\n"))
print(unicodedata.numeric("Ⅳ"))
print(unicodedata.numeric("½"))
print(unicodedata.numeric("7"))
print(unicodedata.decimal("7"))
print(unicodedata.digit("²"))
