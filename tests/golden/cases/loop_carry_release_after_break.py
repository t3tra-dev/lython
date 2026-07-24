# A value carried round a loop and read after `break`. The release inside the
# loop body discharges the PREVIOUS iteration's object, which the back edge
# renamed onto the loop's block argument; the read after the loop consumes the
# current one. The ownership verifier used to conflate the two through the
# resource's un-renamed aliasing views and reject the program.
i = 0
while True:
    i += 1
    if i >= 5:
        break
print(i)

total = 0
n = 0
while True:
    n += 1
    total += n * n
    if n >= 4:
        break
print(n, total)

# `break` out of a bounded loop, same carry shape.
j = 0
while j < 100:
    j += 2
    if j >= 7:
        break
print(j)

# Carried string, released and rebound each iteration.
text = ""
k = 0
while True:
    text = text + str(k)
    k += 1
    if k >= 4:
        break
print(text)
print(len(text))


def in_function(limit: int) -> int:
    count = 0
    while True:
        count += 3
        if count >= limit:
            break
    return count


print(in_function(10))
print(in_function(1))

# Loop that never breaks still releases its carry.
m = 0
while m < 3:
    m += 1
print(m)
