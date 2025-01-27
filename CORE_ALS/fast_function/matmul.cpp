#include <RcppArmadillo.h>
#include "matmul.h"
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;


// [[Rcpp::export]]
mat Matmul(const mat& A, const mat& B) {
  size_t i, j, k;
  size_t A_rows, A_cols, B_cols;
  A_rows = A.n_rows;
  B_cols = B.n_cols;
  A_cols = A.n_cols;
  mat C(A_rows, B_cols, fill::zeros);
  
#pragma omp parallel for
  for (i = 0; i < A_rows; ++i) {
    for  (j = 0; j < A_cols; ++j){
      for (k = 0; k < B_cols; ++k) {
        C(i, k) += A(i, j) * B(j, k);
      }
    }
  }
  return C;
}


// [[Rcpp::export]]
mat Matmul_spar(const mat& A, const mat& B) {
  size_t i, j, k;
  size_t A_rows, A_cols, B_cols;
  A_rows = A.n_rows;
  B_cols = B.n_cols;
  A_cols = A.n_cols;
  mat C(A_rows, B_cols, fill::zeros);
  
#pragma omp parallel for
  for (i = 0; i < A_rows; ++i) {
    for  (j = 0; j < A_cols; ++j){
      if(A(i,j) != 0){
        for (k = 0; k < B_cols; ++k) {
          C(i, k) += A(i, j) * B(j, k);
        }
      }
    }
  }
  return C;
}


// [[Rcpp::export]]

arma::rowvec find_core(const arma::mat& A, int s) {
  size_t num_rows = A.n_rows;
  arma::rowvec result(num_rows);
if(s==0){
  return arma::zeros<arma::rowvec>(num_rows);
}
#pragma omp parallel for
  for (size_t i = 0; i < num_rows; ++i) {
    arma::rowvec row = abs(A.row(i));
    std::nth_element(row.begin(), row.begin() + s - 1, row.end(), std::greater<double>());
    result(i) = row(s - 1);
  }
  
  return result;
}

// [[Rcpp::export]]
arma::mat core_spar(const mat& A, const mat& B, const rowvec& coreindex) {
  size_t i, j, k;
  size_t A_rows, A_cols, B_cols;
  A_rows = A.n_rows;
  B_cols = B.n_cols;
  A_cols = A.n_cols;
  mat C(A_rows, B_cols, fill::zeros);

#pragma omp parallel for
  for (i = 0; i < A_rows; ++i) {
    for  (j = 0; j < A_cols; ++j){
      if(A(i,j) != 0 && abs(A(i,j)) >= coreindex(i)){
        for (k = 0; k < B_cols; ++k) {
          C(i, k) += A(i, j) * B(j, k);
        }
      }
    }
  }
  return C;
}