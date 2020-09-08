## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

## -------------------------------------------------------------------------
def ReadData( filename ):
  numbers = []
  with open( filename ) as f:
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
  n = int( numbers[ 0 ] )
  m = int( numbers[ 1 ] )
  X = []
  Y = []
  for i in range( 2, len( numbers ), n + 1 ):
    x = []
    for j in range( i, i + n ):
      x.append( numbers[ j ] )
    # end for
    X.append( x )
    Y.append( numbers[ i + n ] )
  # end for
  return [ X, Y ]
# end def

## eof - ReadData.py
