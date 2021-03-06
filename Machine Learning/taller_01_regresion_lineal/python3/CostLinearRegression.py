## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy

## -------------------------------------------------------------------------
"""A class to represent the cost of a linear regression problem"""
class CostLinearRegression:

  '''
  Creator method
  @input X input examples as a numpy matrix of m x n dimensions
  @input Y input results as a numpy matrix of m x 1 dimensions
  @output An object created with some intermediary values useful for
          analytic solutions and gradient descent.
  '''
  def __init__( self, X, Y ):

    # Check input validity
    assert ( X.shape[ 0 ] == Y.shape[ 0 ] ), "Invalid input sizes"

    # Copy data
    self.m_M = X.shape[ 0 ]
    self.m_N = X.shape[ 1 ]
    self.m_X = X
    self.m_Y = Y

    # Copy intermediary data
    self.m_XX = numpy.zeros( ( self.m_N, self.m_N ) )
    for i in range( self.m_M ):
      self.m_XX += self.m_X[ i, : ].T @ self.m_X[ i, : ]
    # end for
    self.m_YX = self.m_Y.T @ self.m_X
    self.m_YY = ( self.m_Y.T @ self.m_Y ).item( )
    self.m_sX = numpy.sum( self.m_X, axis = 0 )
    self.m_sY = numpy.sum( self.m_Y )
  # end def __init__

  '''
  Evaluation method
  @input w parameter 1 x n vector
  @input b bias real number
  @output J(w,b)
  '''
  def evaluate( self, w, b ):
    J = \
      ( ( w @ self.m_XX @ w.T ) ).item( ) + \
      ( ( 2.0 * b ) * ( w @ self.m_sX.T ).item( ) ) + \
      ( b * b * float( self.m_M ) ) - \
      ( 2.0 * ( w @ self.m_YX.T ).item( ) ) - \
      ( 2.0 * b * self.m_sY ) + \
      self.m_YY
    return( J / float( self.m_M ) )
  # end def eval

  '''
  Gradient calculation method
  @input w parameter 1 x n vector
  @input b bias real number
  @output [ dJ(w,b)/dw, dJ(w,b)/db ]
  '''
  def gradient( self, w, b ):
    dof = 2.0 / float( self.m_M )
    dw = ( w @ self.m_XX ) + ( b * self.m_sX ) - self.m_YX
    db = \
       ( w @ self.m_sX.T ).item( ) + \
       ( b * float( self.m_M ) ) - \
       self.m_sY
    return [ dw / dof, db / dof ]
  # end def gradient

  '''
  Analytic solution of the linear regression problem
  @output [ w, b ] real parameters that minimize the problem given the
          inputs X and Y.
  '''
  def analytic_solve( self ):
    w = \
      ( self.m_YX - ( self.m_sX * ( self.m_sY / float( self.m_M ) ) ) ) @ \
      numpy.linalg.inv( \
        ( self.m_X.T @ self.m_X ) - \
        ( ( self.m_sX.T @ self.m_sX ) / float( self.m_M ) ) \
        )
    b = ( self.m_sY - ( w @ self.m_sX.T ).item( ) ) / float( self.m_M )
    return [ w, b.sum( ) ]
  # end def analytic_solve

  def w_shape( self ):
    return ( 1, self.m_N )
  # end def w_shape

# end class CostLinearRegression

## eof - CostLinearRegression.py
