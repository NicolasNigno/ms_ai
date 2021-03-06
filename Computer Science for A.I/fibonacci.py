def getLastTwo(x):
    if x <= 1:
        return x
    else:
        return (getLastTwo(x-1) + getLastTwo(x-2))

print('Cantidad de números de Fibonacci:')
number = int(input())

for i in range(number):
    print(getLastTwo(i))