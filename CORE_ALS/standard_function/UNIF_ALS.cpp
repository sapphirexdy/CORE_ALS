#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "matmul.h"
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(Rcpp)]]
using namespace arma;


// [[Rcpp::export]]
Rcpp::List unif(const arma::mat& R, double lambda_, int dim_factors, int n_iter, double r) {
  arma::mat Bool_matrix = R;
  Bool_matrix.replace(datum::nan, 0); // Replace NaN with 0
  Bool_matrix.transform([](double val) { return val != 0 ? 1.0 : 0.0; }); // Convert to binary
  arma::mat result_matrix = Bool_matrix;
  arma::vec row_sums = sum(result_matrix, 1);
  arma::vec col_sums = sum(result_matrix, 0).t();
  
  int n_u = R.n_rows;
  int n_m = R.n_cols;
  
  // Initialize U and M
  arma::mat U = randu<arma::mat>(dim_factors, n_u); // Random initialization
  arma::mat M = randu<arma::mat>(dim_factors, n_m);
  
  arma::mat lamI = lambda_ * arma::eye<arma::mat>(dim_factors, dim_factors);
  
  for (int i_iter = 0; i_iter < n_iter; ++i_iter) {
    // Update M
  #pragma omp parallel for
    for (int m = 0; m < n_m; ++m) {
      arma::uvec users = find(Bool_matrix.col(m)); // Non-zero user indices
      //std::cout<< users.n_elem << std::endl;
      if (users.n_elem > 0) {
        arma::mat Um = U.cols(users);
        arma::uvec sample_U = randperm(users.n_elem, std::round(r * users.n_elem));
        arma::mat R_u = R.col(m);
        arma::vec vector = row_spar(Um , R_u.rows(users),sample_U);
        arma::mat matrix = row_spar(Um , Um.t(),sample_U); + col_sums(m) * lamI;
        M.col(m) = solve(matrix, vector);
      }
    }
    
    // Update U
  #pragma omp parallel for
    for (int u = 0; u < n_u; ++u) {
      arma::uvec items = find(Bool_matrix.row(u)); // Non-zero item indices
      if (items.n_elem > 0) {
        arma::mat Mu = M.cols(items);
        arma::uvec sample_M = randperm(items.n_elem, std::round(r * items.n_elem));
        arma::mat R_m = R.row(u);
        arma::vec vector = row_spar(Mu , R_m.cols(items).t(),sample_M);
        arma::mat matrix = row_spar(Mu , Mu.t(),sample_M) + row_sums(u) * lamI;
        U.col(u) = solve(matrix, vector);
      }
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("U") = U.t(), Rcpp::Named("M") = M);
}
