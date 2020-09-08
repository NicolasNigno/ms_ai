// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#ifndef __CostLinearRegression__h__
#define __CostLinearRegression__h__

#include <istream>
#include <Eigen/Core>

/**
 * A class to represent the cost of a linear regression problem
 */
template< class _TScalar >
class CostLinearRegression
{
public:
  using Self    = CostLinearRegression;
  using TScalar = _TScalar;
  using TMatrix = Eigen::Matrix< TScalar, Eigen::Dynamic, Eigen::Dynamic >;

public:
  CostLinearRegression( );
  virtual ~CostLinearRegression( ) = default;

  const unsigned long& size( ) const;
  const unsigned long& number_of_samples( ) const;

  /*
   * Evaluation method
   * @input w parameter 1 x n vector
   * @input b bias real number
   * @output J(w,b)
   */
  TScalar evaluate( const TMatrix& w, const TScalar& b ) const;

  /*
   * Gradient calculation method
   * @input w parameter 1 x n vector
   * @input b bias real number
   * @output [ dJ(w,b)/dw, dJ(w,b)/db ]
   */
  void gradient(
    const TMatrix& w, const TScalar& b, TMatrix& dw, TScalar& db
    ) const;

  /*
   * Analytic solution of the linear regression problem
   * @output [ w, b ] real parameters that minimize the problem given the
   *         inputs X and Y.
   */
  void analytic_solve( TMatrix& w, TScalar& b );

  /*
   * Creator method
   * @input X input examples as a numpy matrix of m x n dimensions
   * @input Y input results as a numpy matrix of m x 1 dimensions
   * @output An object created with some intermediary values useful for
   *         analytic solutions and gradient descent.
   */
  void read( std::istream& input );

private:
  CostLinearRegression( const Self& other ) = delete;
  CostLinearRegression& operator=( const Self& other ) = delete;

protected:
  TMatrix m_X;
  TMatrix m_Y;
  TMatrix m_XX;
  TMatrix m_YX;
  TMatrix m_sX;
  TScalar m_YY;
  TScalar m_sY;

  unsigned long m_M;
  unsigned long m_N;

public:
  friend std::istream& operator>>( std::istream& in, Self& cost )
    {
      cost.read( in );
      return( in );
    }
};

#include "CostLinearRegression.hxx"

#endif // __CostLinearRegression__h__

// eof - CostLinearRegression.h
