import sys, time
import numpy
from ReadData import ReadData
from CostLinearRegression import CostLinearRegression

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

# 3. Analytical version
sT = time.process_time( )

linearRegression = CostLinearRegression( X, Y )
[w, b] = linearRegression.analytic_solve()
J = linearRegression.evaluate(w, b)
#[ w, b, J ] = AnalyticalLinearRegression( X, Y )
eT = time.process_time( )
print( "Analytical version done in", float( eT - sT ), "seconds" )
print( "----------------------------" )
print( "---- Analytical results ----" )
print( "----------------------------" )
print( "w =", w )
print( "b =", b )
print( "J =", J )
print( "----------------------------" )

## eof - analytical_linear_regression.py
