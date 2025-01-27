#ifndef MATMUL_H
#define MATMUL_H

#include <RcppArmadillo.h>

// 声明函数
arma::mat Matmul(const arma::mat& A, const arma::mat& B);
arma::mat Matmul_spar(const arma::mat& A, const arma::mat& B);
arma::rowvec find_core(const arma::mat& A, int s);
arma::mat core_spar(const arma::mat& A, const arma::mat& B, const arma::rowvec& coreindex);
arma::mat row_spar(const arma::mat& A, const arma::mat& B, const arma::uvec& sampleindex);
#endif // MATMUL_H
