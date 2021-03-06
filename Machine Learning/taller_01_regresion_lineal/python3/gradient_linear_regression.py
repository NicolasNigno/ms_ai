import sys, time
import numpy
#import matplotlib.pyplot as plt
from ReadData import ReadData
from CostLinearRegression import CostLinearRegression
#from AnalyticalLinearRegression import *
#from GradientDescentLinearRegression import *

# 1. Read data
data_path = 'p_04.txt'
sT = time.process_time( )
[ X_data, Y_data ] = ReadData(sys.argv[ 1 ])
eT = time.process_time( )
print( "Data read from", sys.argv[ 1 ], "in", float( eT - sT ), "seconds" )

# 2. Create numpy objects
sT = time.process_time( )
X = numpy.matrix( X_data )
Y = numpy.matrix( Y_data ).transpose( )
eT = time.process_time( )
print( "Data converted in", float( eT - sT ), "seconds" )

# 3. Gradient descent version
sT = time.process_time( )

c = CostLinearRegression( X, Y )
w = numpy.zeros( c.w_shape( ) )
b = 0.0
J = c.evaluate( w, b )
print(J)

# Main loop
alpha = float(sys.argv[ 2 ])
epsilon = float(sys.argv[ 3 ])
i = 0
stop = False
while not stop:

  # Update parameters
  [ dw, db ] = c.gradient( w, b )
  w -= dw * alpha
  b -= db * alpha
  Jn = c.evaluate( w, b )
  #print(Jn)

  # Check termination
  if J - Jn < epsilon:
    stop = True
  # end if
  i = i + 1
  if i % 10000 == 0:
    print( "\33[2K\rIteration:", i, "dJ =", J - Jn, end = "" )
  # end if
  J = Jn
# end while

eT = time.process_time( )
print( "" )
print( "Gradient descent version done in", float( eT - sT ), "seconds" )
print( "----------------------------------" )
print( "---- Gradient descent results ----" )
print( "----------------------------------" )
print( "w =", w )
print( "b =", b )
print( "J =", J )
print( "Iterations =", i )
print( "----------------------------------" )

## eof - gradient_linear_regression.py
