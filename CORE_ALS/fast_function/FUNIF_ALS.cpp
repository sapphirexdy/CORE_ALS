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
arma::mat sample_rows(const arma::mat& R, double r) {
  // 确定需要抽取的行数
  int n_rows = R.n_rows;
  int n_sampled_rows = std::round(r * n_rows);
  
  // 创建一个0-1矩阵，用来标记哪些行需要被保留
  arma::uvec sampled_indices = randperm(n_rows, n_sampled_rows); // 抽取索引
  
  // 创建结果矩阵，将其初始化为0
  arma::mat result = arma::zeros<arma::mat>(n_rows, R.n_cols);
  
  // 保留抽取的行
  result.rows(sampled_indices) = R.rows(sampled_indices);
  
  return result;
}

// [[Rcpp::export]]
arma::mat sample_cols(const arma::mat& R, double r) {
  // 确定需要抽取的列数
  int n_cols = R.n_cols;
  int n_sampled_cols = std::round(r * n_cols);
  
  // 创建一个0-1矩阵，用来标记哪些列需要被保留
  arma::uvec sampled_indices = randperm(n_cols, n_sampled_cols); // 抽取索引
  
  // 创建结果矩阵，将其初始化为0
  arma::mat result = arma::zeros<arma::mat>(R.n_rows, n_cols);
  
  // 保留抽取的列
  result.cols(sampled_indices) = R.cols(sampled_indices);
  
  return result;
} 

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
  arma::mat Bool_matrixU= sample_rows(Bool_matrix,r);
  #pragma omp parallel for
    for (int m = 0; m < n_m; ++m) {
      arma::uvec users = find(Bool_matrixU.col(m)); // Non-zero user indices
      //std::cout<< users.n_elem << std::endl;
      if (users.n_elem > 0) {
        arma::mat Um = U.cols(users);
        arma::mat R_u = R.col(m);
        arma::vec vector = Matmul(Um , R_u.rows(users));
        arma::mat matrix = Matmul(Um , Um.t()) + col_sums(m) * lamI;
        M.col(m) = solve(matrix, vector);
      }
    }
    
    // Update U
  arma::mat Bool_matrixM= sample_cols(Bool_matrix,r);
  #pragma omp parallel for
    for (int u = 0; u < n_u; ++u) {
      arma::uvec items = find(Bool_matrixM.row(u)); // Non-zero item indices
      if (items.n_elem > 0) {
        arma::mat Mu = M.cols(items);
        arma::mat R_m = R.row(u);
        arma::vec vector = Matmul(Mu , R_m.cols(items).t());
        arma::mat matrix = Matmul(Mu , Mu.t()) + row_sums(u) * lamI;
        U.col(u) = solve(matrix, vector);
      }
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("U") = U.t(), Rcpp::Named("M") = M);
}
