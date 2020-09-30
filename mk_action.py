import itertools

for i in itertools.product('01', repeat=6):
    print(" ".join(i))

