## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, random, sys
import numpy
import matplotlib.pyplot as plt

# Check command line options
if len( sys.argv ) < 6:
  print(
    "Usage: " + sys.argv[ 0 ] +
    " out_fname x_min x_max m r ai=float aj=float ..."
    )
  sys.exit( 1 )
# end if

# Get command line values
out_fname = sys.argv[ 1 ]
x_min = float( sys.argv[ 2 ] )
x_max = float( sys.argv[ 3 ] )
m = int( sys.argv[ 4 ] )
r = float( sys.argv[ 5 ] )
wb = []
for arg in sys.argv[ 6 : ]:
  i = int( arg[ 1 : arg.find( '=' ) ] )
  if len( wb ) < i + 1:
    wb += [ 0 for i in range( i + 1 - len( wb ) ) ]
  # end if
  wb[ i ] = float( arg[ arg.find( '=' ) + 1 : ] )
  # end if
# end for

# Create numpy objects
n = len( wb ) - 1
w = numpy.array( wb[ 1 : ] ).reshape( ( n, 1 ) )
b = wb[ 0 ]

# Create data
X = numpy.zeros( ( m, n ) )
for i in range( m ):
  v = ( ( float( i ) / float( m - 1 ) ) * ( x_max - x_min ) ) + x_min
  X[ i ] = numpy.array(
    [ [ v ** i ] for i in range( 1, w.shape[ 0 ] + 1 ) ]
  ).reshape( ( 1, n ) )
# end for
Y = numpy.dot( X, w ) + b

# Plot real polinomial data
# fig, axs = plt.subplots( 1, 1 )
# axs.plot( X[ :, [ 0 ] ], Y[ :, [ 0 ] ] )
# axs.grid( )
# axs.axvline( )
# axs.axhline( )
 
# Insert some noise
if r > 0:
  D = numpy.append( X[ :, [ 0 ] ], Y[ :, [ 0 ] ], axis = 1 )
  E = D[ : -1, : ] - D[ 1 : , : ]
  d = numpy.sum( numpy.sqrt( E[ :, [ 0 ] ] ** 2 + E[ :, [ 1 ] ] ** 2 ) )
  d *= r
  angles = numpy.random.rand( m, 1 ) * ( 2 * math.pi )
  radii = ( numpy.random.rand( m, 1 ) - 0.5 ) * ( 2 * d )
  X[ :, [ 0 ] ] += radii * numpy.cos( angles )
  Y[ :, [ 0 ] ] += radii * numpy.sin( angles )
# end if
D = numpy.append( X, Y, axis = 1 )
numpy.random.shuffle( D )

# Write result
out_file = open( out_fname, "w" )
print( "{:d} {:d}".format( D.shape[ 1 ] - 1, D.shape[ 0 ] ), file = out_file )
for i in range( D.shape[ 0 ] ):
  for j in range( D.shape[ 1 ] ):
    print( "{:.4f} ".format( D[ i ][ j ] ), end = "", file = out_file )
  # end for
  print( "", file = out_file )
# end for
out_file.close( )

# Plot scattered data
# axs.scatter( X[ :,[ 0 ] ], Y[ :,[ 0 ] ], s = 1, c = "#ff0000" )
# TODO: axs.set_aspect( "equal", "box" )
# plt.show( )

## eof - create_polynomial_data.py
