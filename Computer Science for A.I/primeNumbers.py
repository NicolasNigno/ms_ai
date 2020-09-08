print('Enter a number: ')
number = input()
assert number.isnumeric(), 'Enter a number'

number = int(number)

def isPrime(n):
    if n == 1:
        return 0
    if n == 2:
        return 1
    if n % 2 == 0:
        return 0
    i = 3
    n = int(n)
    for i in range(3, int(n/2), 2):
        if n % i == 0:
            return 0
    return 1

print(isPrime(number))