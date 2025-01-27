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
arma::uvec sample_cols(const int n_cols, const arma::vec& p, int r) {
  
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
  
  
  return selected_cols;
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
#pragma omp parallel for
    for (int m = 0; m < n_m; ++m) {
      arma::uvec users = find(Bool_matrix.col(m)); // Non-zero user indices
      if (users.n_elem > 0) {
        arma::mat Um = U.cols(users);
        arma::vec pu = RowProb(Um.t());
        arma::uvec sample_U = sample_cols(users.n_elem, pu,std::round(r * users.n_elem));
        arma::mat R_u = R.col(m);
        arma::vec vector = row_spar(Um , R_u.rows(users),sample_U);
        arma::mat matrix = row_spar(Um , Um.t(),sample_U) + col_sums(m) * lamI;
        M.col(m) = solve(matrix, vector);
      }
    }
    
    // Update U
#pragma omp parallel for
    for (int u = 0; u < n_u; ++u) {
      arma::uvec items = find(Bool_matrix.row(u)); // Non-zero item indices
      if (items.n_elem > 0) {
        arma::mat Mu = M.cols(items);
        arma::vec pm = RowProb(Mu.t());
        arma::uvec sample_M = sample_cols(items.n_elem, pm,std::round(r * items.n_elem));
        arma::mat R_m = R.row(u);
        arma::vec vector = row_spar(Mu , R_m.cols(items).t(),sample_M);
        arma::mat matrix = row_spar(Mu , Mu.t(),sample_M) + row_sums(u) * lamI;
        U.col(u) = solve(matrix, vector);
      }
    }
  }
  
  return Rcpp::List::create(Rcpp::Named("U") = U.t(), Rcpp::Named("M") = M);
}
