
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

  if( argc < 2 )
  {
    std::cerr << "Usage: " << argv[ 0 ] << " filename" << std::endl;
    return( 1 );
  } // end if
  std::string filename = argv[ 1 ];

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

  // Solve analytically
  TCost::TMatrix w;
  TCost::TScalar b;
  sT = std::chrono::high_resolution_clock::now( );
  c.analytic_solve( w, b );
  eT = std::chrono::high_resolution_clock::now( );
  rT =
    std::chrono::duration_cast< std::chrono::nanoseconds >( eT - sT ).
    count( );
  rT *= 10e-10;

  std::cout
    << "Analytical version done in " << rT << " seconds" << std::endl
    << "----------------------------" << std::endl
    << "---- Analytical results ----" << std::endl
    << "----------------------------" << std::endl
    << "w = " << w << std::endl
    << "b = " << b << std::endl
    << "J = " << c.evaluate( w, b ) << std::endl
    <<  "----------------------------" << std::endl;

  return( 0 );
}

// eof - analytical_linear_regression.cxx
