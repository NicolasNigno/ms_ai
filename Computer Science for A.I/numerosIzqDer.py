print('Ingresa un número de hasta 6 dígitos')
number = int(input()) ## Si no es un número bota error al hacer el cast a int

number = str(number) ## se convierte en string el número
assert len(number)<=6, 'el número solo puede tener 6 dígitos'
print(number[::-1])