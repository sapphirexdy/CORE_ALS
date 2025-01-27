#include <RcppArmadillo.h>
#include <Rcpp.h>
#include "matmul.h"
#ifdef _OPENMP
#include <omp.h>
#endif
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(Rcpp)]]
using namespace arma;

#include <RcppArmadillo.h>
#include <Rcpp.h>
using namespace arma;

// [[Rcpp::export]]
arma::mat sample_rows(const arma::mat& R, const arma::vec& p, int r) {
  int n_rows = R.n_rows;
  int n_cols = R.n_cols;
  
  // 创建结果矩阵，将其初始化为0
  arma::mat result = arma::zeros<arma::mat>(n_rows, n_cols);
  
  // 检查概率数组的长度是否与矩阵行数相符
  if (p.n_elem != n_rows) {
    Rcpp::stop("The length of probability array p must match the number of rows in R.");
  }
  
  // 创建一个随机数生成器并初始化
  arma::uvec selected_rows = arma::zeros<arma::uvec>(r);
  
  // 抽取行的过程
  int selected_count = 0;
  while (selected_count < r) {
    // 随机生成一个[0,1)的数值，并检查它是否小于相应的概率p[i]
    for (int i = 0; i < n_rows; ++i) {
      if (randu() < p[i]) { // 如果小于概率值p[i]
        // 如果当前行还未被选择，选择该行
        if (std::find(selected_rows.begin(), selected_rows.end(), i) == selected_rows.end()) {
          selected_rows[selected_count] = i;
          selected_count++;
          if (selected_count >= r) break;
        }
      }
    }
  }
  
  // 根据选中的行，将对应的行从原矩阵R复制到结果矩阵中
  for (int i = 0; i < r; ++i) {
    result.row(selected_rows[i]) = R.row(selected_rows[i]);
  }
  
  return result;
}


// [[Rcpp::export]]
arma::mat sample_cols(const arma::mat& R, const arma::vec& p, int r) {
  int n_rows = R.n_rows;
  int n_cols = R.n_cols;
  
  // 创建结果矩阵，将其初始化为0
  arma::mat result = arma::zeros<arma::mat>(n_rows, n_cols);
  
  // 检查概率数组的长度是否与矩阵列数相符
  if (p.n_elem != n_cols) {
    Rcpp::stop("The length of probability array p must match the number of columns in R.");
  }
  
  // 创建一个随机数生成器并初始化
  arma::uvec selected_cols = arma::zeros<arma::uvec>(r);
  
  // 抽取列的过程
  int selected_count = 0;
  while (selected_count < r) {
    // 随机生成一个[0,1)的数值，并检查它是否小于相应的概率p[i]
    for (int i = 0; i < n_cols; ++i) {
      if (randu() < p[i]) { // 如果小于概率值p[i]
        // 如果当前列还未被选择，选择该列
        if (std::find(selected_cols.begin(), selected_cols.end(), i) == selected_cols.end()) {
          selected_cols[selected_count] = i;
          selected_count++;
          if (selected_count >= r) break;
        }
      }
    }
  }
  
  // 根据选中的列，将对应的列从原矩阵R复制到结果矩阵中
  for (int i = 0; i < r; ++i) {
    result.col(selected_cols[i]) = R.col(selected_cols[i]);
  }
  
  return result;
}


// [[Rcpp::export]]
arma::vec RowProb(const arma::mat& X) {
  arma::vec p;
  arma::mat U, V;
  arma::vec s;
  arma::svd_econ(U, s, V, X);
  p = arma::sum(arma::square(U), 1);
  p = p / arma::sum(p);
  return p;
}

// [[Rcpp::export]]
Rcpp::List lev(const arma::mat& R, double lambda_, int dim_factors, int n_iter, double r) {
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
    arma::vec pu = RowProb(U.t());
    arma::mat Bool_matrixU= sample_rows(Bool_matrix,pu,std::round(r * n_u));
#pragma omp parallel for
    for (int m = 0; m < n_m; ++m) {
      arma::uvec users = find(Bool_matrixU.col(m)); // Non-zero user indices
      if (users.n_elem > 0) {
        arma::mat Um = U.cols(users);
        arma::mat R_u = R.col(m);
        arma::vec vector = Matmul(Um , R_u.rows(users));
        arma::mat matrix = Matmul(Um , Um.t()) + col_sums(m) * lamI;
        M.col(m) = solve(matrix, vector);
      }
    }
    
    // Update U
    arma::vec pm = RowProb(M.t());
    arma::mat Bool_matrixM= sample_cols(Bool_matrix,pm,std::round(r * n_m));
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
