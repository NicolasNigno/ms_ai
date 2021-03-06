## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
##
## @execution
## python3 AnalyticLinearRegression.py <file_with_data>
## =========================================================================

import sys, time

## -------------------------------------------------------------------------
def Solve( X, Y ):
  # Check input validity
  assert ( len( X ) == len( Y ) ), "Invalid input sizes"

  # Cached values
  m = len( X )
  sx = 0.0
  sy = 0.0
  sx2 = 0.0
  sxy = 0.0
  for i in range( m ):
    sx += X[ i ]
    sy += Y[ i ]
    sx2 += X[ i ] * X[ i ]
    sxy += X[ i ] * Y[ i ]
  # end for

  # Weight and bias calculation
  A = ( float( m ) * sx2 ) - ( sx * sx )
  w = ( ( float( m ) * sxy ) - ( sx * sy ) ) / A
  b = ( ( sx2 * sy ) - ( sx * sxy ) ) / A

  # Cost
  J = 0.0
  for i in range( m ):
    J += ( ( w * X[ i ] ) + b - Y[ i ] ) ** 2
  # end for
  J /= float( m )

  return [ w, b, J ]
# end def

## -------------------------------------------------------------------------
## "Main" function
## -------------------------------------------------------------------------

# Check arguments
if len( sys.argv ) != 2:
  print( "Usage: ", sys.argv[ 0 ], "input_data_file" )
  exit( 1 )
# end if
filename = sys.argv[ 1 ]

# Read data
sT = time.process_time( )
numbers = []
with open( sys.argv[ 1 ] ) as f:
  for line in f:
    for i in line.split( ):
      try:
        v = float( i )
        numbers.append( v )
      except ValueError:
        pass
      # end try
    # end for
  # end for
# end with
m = int( numbers[ 0 ] )
n = int( numbers[ 1 ] )
X = []
Y = []
for i in range( 2, len( numbers ), 2 ):
  X.append( numbers[ i ] )
  Y.append( numbers[ i + 1 ] )
# end for
eT = time.process_time( )
print( "Data read in", float( eT - sT ), "seconds." )

# Solve regression
sT = time.process_time( )
[ w, b, J ] = Solve( X, Y )
eT = time.process_time( )
print( "Regression solved in", float( eT - sT ), "seconds." )

print( "w =", w )
print( "b =", b )
print( "J =", J )

## eof - AnalyticLinearRegression.py
