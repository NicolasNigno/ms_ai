## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math, random, sys

# Check command line options
if len( sys.argv ) < 4:
  print( "Usage: " + sys.argv[ 0 ] + " out_fname n m" )
  sys.exit( 1 )
# end if

# Get command line values
out_fname = sys.argv[ 1 ]
n = int( sys.argv[ 2 ] )
m = int( sys.argv[ 3 ] )

# Write result
out_file = open( out_fname, "w" )
print( "{:d} {:d}".format( n, m ), file = out_file )
for i in range( m ):
  for j in range( n + 1 ):
    print( "{:.4f} ".format( random.uniform( -1000, 1000 ) ), end = "", file = out_file )
  # end for
  print( "", file = out_file )
# end for
out_file.close( )

## eof - create_random_data.py
