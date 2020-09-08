import math
import cmath

print("Enter value a: ")
a = input()
a = float(a)
assert a != 0, 'Make sure a is not 0'

print("Enter value b: ")
b = input()
b = float(b)

print('Enter value c:')
c = input()
c = float(c)

def quadratic(a, b, c):
    if b**2 >= 4*a*c:
        x1 = (-b + math.sqrt(b**2 - 4*a*c))/(2*a)
        x2 = (-b - math.sqrt(b**2 - 4*a*c))/(2*a)
        
        return x1, x2
    elif b**2 < 4*a*c:
        x1 = (-b + cmath.sqrt(b**2 - 4*a*c))/(2*a)
        x2 = (-b - cmath.sqrt(b**2 - 4*a*c))/(2*a)
        
        return x1, x2
    
print(quadratic(a, b, c))