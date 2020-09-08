def denominador(a, b):
    while b != 0:
        (a, b) = (b, a % b)
    return a

print('El máximo común denominador es: %s' %denominador(54, 24))