#include <RcppArmadillo.h>
#include "matmul.h"
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::depends(RcppArmadillo)]]

using namespace arma;



// [[Rcpp::export]]
Rcpp::List FCORE(const arma::mat& R, double lambda_, int dim_factors, int n_iter, double r) {
  // Convert to binary matrix
  arma::mat Bool_matrix = R;
  Bool_matrix.replace(datum::nan, 0); // Replace NaN with 0
  Bool_matrix.transform([](double val) { return val != 0 ? 1.0 : 0.0; }); // Convert to binary
  
  arma::vec row_sums = sum(Bool_matrix, 1);
  arma::vec col_sums = sum(Bool_matrix, 0).t();
  
  int n_u = R.n_rows;
  int n_m = R.n_cols;
  
  // Initialize U and M
  arma::mat U = randu<arma::mat>(dim_factors, n_u); // Random initialization
  arma::mat M = randu<arma::mat>(dim_factors, n_m);
  
  arma::mat lamI = lambda_ * arma::eye<arma::mat>(dim_factors, dim_factors);
  
  for (int i_iter = 0; i_iter < n_iter; ++i_iter) {

    // Update M
  arma::rowvec coreindex_U = find_core(U, std::round(r * n_u));
  #pragma omp parallel for
      for (int m = 0; m < n_m; ++m) {
        uvec users = find(Bool_matrix.col(m));// Non-zero user indices
        if (users.n_elem > 0) {
          arma::mat Um = U.cols(users);
          
          //std::cout<< std::round(r * users.n_elem) << std::endl;
          arma::mat R_u = R.col(m);
          arma::vec vector = core_spar(Um , R_u.rows(users), coreindex_U);
          arma::mat matrix = core_spar(Um , Um.t(), coreindex_U) + col_sums(m) *lamI;
          M.col(m) = solve(matrix, vector);
        }
      }
    
    
    // Update U
  arma::rowvec coreindex_M = find_core(M, std::round(r*n_m));
  #pragma omp parallel for
      for (int u = 0; u < n_u; ++u) {
        uvec items = find(Bool_matrix.row(u));  // Non-zero item indices
        if (items.n_elem > 0) {
          arma::mat Mu = M.cols(items);
          
          arma::mat R_m = R.row(u);
          arma::vec vector = core_spar(Mu , R_m.cols(items).t() , coreindex_M);
          arma::mat matrix = core_spar(Mu , Mu.t(), coreindex_M) + row_sums(u) *lamI;
          U.col(u) = solve(matrix, vector);
        }
      }
    }
  
  return Rcpp::List::create(Rcpp::Named("U") = U.t(), Rcpp::Named("M") = M);
}

