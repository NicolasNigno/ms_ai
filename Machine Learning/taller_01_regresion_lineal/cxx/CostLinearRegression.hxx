// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#ifndef __CostLinearRegression__hxx__
#define __CostLinearRegression__hxx__

#include <Eigen/Dense>

// -------------------------------------------------------------------------
template< class _TScalar >
CostLinearRegression< _TScalar >::
CostLinearRegression( )
  : m_M( 0 ),
    m_N( 0 )
{
}

// -------------------------------------------------------------------------
template< class _TScalar >
const unsigned long& CostLinearRegression< _TScalar >::
size( ) const
{
  return( this->m_N );
}

// -------------------------------------------------------------------------
template< class _TScalar >
const unsigned long& CostLinearRegression< _TScalar >::
number_of_samples( ) const
{
  return( this->m_M );
}

// -------------------------------------------------------------------------
template< class _TScalar >
_TScalar CostLinearRegression< _TScalar >::
evaluate( const TMatrix& w, const TScalar& b ) const
{
  TScalar J = ( w * this->m_XX * w.transpose( ) )( 0, 0 );
  J += TScalar( 2 ) * b * ( w * this->m_sX.transpose( ) )( 0, 0 );
  J += b * b * TScalar( this->m_M );
  J -= TScalar( 2 ) * ( w * this->m_YX.transpose( ) )( 0, 0 );
  J -= TScalar( 2 ) * b * this->m_sY;
  J += this->m_YY;
  return( J / TScalar( this->m_M ) );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void CostLinearRegression< _TScalar >::
gradient( const TMatrix& w, const TScalar& b, TMatrix& dw, TScalar& db ) const
{
  TScalar dof = TScalar( 2 ) / TScalar( this->m_M );
  dw = ( ( w * this->m_XX ) + ( b * this->m_sX ) - this->m_YX ) / dof;
  db  = ( w * this->m_sX.transpose( ) )( 0, 0 );
  db += b * TScalar( this->m_M );
  db -= this->m_sY;
  db /= dof;
}

// -------------------------------------------------------------------------
template< class _TScalar >
void CostLinearRegression< _TScalar >::
analytic_solve( TMatrix& w, TScalar& b )
{
  TMatrix v =
    this->m_YX - ( this->m_sX * ( this->m_sY / TScalar( this->m_M ) ) );
  TMatrix A =
    ( this->m_X.transpose( ) * this->m_X ) -
    ( ( this->m_sX.transpose( ) * this->m_sX ) / TScalar( this->m_M ) );
  w = v * A.inverse( );
  b =
    ( this->m_sY - ( w * this->m_sX.transpose( ) )( 0, 0 ) ) /
    TScalar( this->m_M );
}

// -------------------------------------------------------------------------
template< class _TScalar >
void CostLinearRegression< _TScalar >::
read( std::istream& input )
{
  input >> this->m_N >> this->m_M;
  this->m_X.resize( this->m_M, this->m_N );
  this->m_Y.resize( this->m_M, 1 );

  for( unsigned long m = 0; m < this->m_M; ++m )
  {
    for( unsigned long n = 0; n < this->m_N; ++n )
      input >> this->m_X( m, n );
    input >> this->m_Y( m, 0 );
  } // end for

  this->m_XX = TMatrix::Zero( this->m_N, this->m_N );
  for( unsigned long i = 0; i < this->m_M; ++i )
    this->m_XX += this->m_X.row( i ).transpose( ) * this->m_X.row( i );

  this->m_YX = this->m_Y.transpose( ) * this->m_X;
  this->m_YY = ( this->m_Y.transpose( ) * this->m_Y )( 0, 0 );
  this->m_sX = this->m_X.colwise( ).sum( );
  this->m_sY = this->m_Y.sum( );
}

#endif // __CostLinearRegression__hxx__

// eof - CostLinearRegression.hxx
