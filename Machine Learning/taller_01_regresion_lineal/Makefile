all: cxx_analytical_linear_regression cxx_gradient_linear_regression

cxx_analytical_linear_regression: cxx/analytical_linear_regression.cxx cxx/CostLinearRegression.h cxx/CostLinearRegression.hxx
	g++ -std=c++17 -Ofast cxx/analytical_linear_regression.cxx -o cxx_analytical_linear_regression -I/usr/local/include/eigen3

cxx_gradient_linear_regression: cxx/gradient_linear_regression.cxx cxx/CostLinearRegression.h cxx/CostLinearRegression.hxx
	g++ -std=c++17 -Ofast cxx/gradient_linear_regression.cxx -o cxx_gradient_linear_regression -I/usr/local/include/eigen3