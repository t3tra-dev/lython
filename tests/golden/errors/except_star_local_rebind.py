n = 0
try:
    raise ExceptionGroup("g", [ValueError("v")])
except* ValueError as eg:
    n += 1
print(n)
