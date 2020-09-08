
#include "CostLinearRegression.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

int main( int argc, char** argv )
{
  using TScalar = long double;
  using TCost   = CostLinearRegression< TScalar >;

  if( argc < 4 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " filename alpha epsilon" << std::endl;
    return( 1 );
  } // end if
  std::string filename = argv[ 1 ];
  std::stringstream params;
  params << argv[ 2 ] << " " << argv[ 3 ];
  std::istringstream iparams( params.str( ) );
  TCost::TScalar alpha, epsilon;
  iparams >> alpha >> epsilon;

  // Read data
  auto sT = std::chrono::high_resolution_clock::now( );
  TCost c;
  std::ostringstream buf_str;
  std::ifstream input_file( filename.c_str( ), std::ios::binary );
  buf_str << input_file.rdbuf( );
  std::istringstream input( buf_str.str( ) );
  input_file.close( );
  input >> c;
  auto eT = std::chrono::high_resolution_clock::now( );
  double rT =
    std::chrono::duration_cast< std::chrono::nanoseconds >( eT - sT ).
    count( );
  rT *= 10e-10;
  std::cout
    << "Data read from " << filename << " in " << rT << " seconds " << std::endl;

  // Gradient descent
  sT = std::chrono::high_resolution_clock::now( );

  TCost::TMatrix w = TCost::TMatrix::Zero( 1, c.size( ) );
  TCost::TScalar b = TCost::TScalar( 0 );
  TCost::TScalar J = c.evaluate( w, b );

  unsigned long i = 0;
  bool stop = false;
  while( !stop )
  {
    // Update parameters
    TCost::TMatrix dw;
    TCost::TScalar db;
    c.gradient( w, b, dw, db );
    w -= dw * alpha;
    b -= db * alpha;
    TCost::TScalar Jn = c.evaluate( w, b );

    // Check termination
    stop = ( J - Jn < epsilon );
    i += 1;
    if( i % 1000 == 0 )
      std::cout
        << "\33[2K\rIteration: " << i
        << ", dJ = " << ( J - Jn )
        << std::flush;
    J = Jn;
  } // end while

  eT = std::chrono::high_resolution_clock::now( );
  rT =
    std::chrono::duration_cast< std::chrono::nanoseconds >( eT - sT ).
    count( );
  rT *= 10e-10;

  std::cout
    << std::endl
    << "Gradient descent version done in " << rT << " seconds" << std::endl
    << "----------------------------------" << std::endl
    << "---- Gradient descent results ----" << std::endl
    << "----------------------------------" << std::endl
    << "w = " << w << std::endl
    << "b = " << b << std::endl
    << "J = " << c.evaluate( w, b ) << std::endl
    << "Iterations = " << i << std::endl
    <<  "---------------------------------" << std::endl;

  return( 0 );
}

// eof - gradient_linear_regression.cxx
