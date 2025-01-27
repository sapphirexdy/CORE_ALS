# Load required libraries
library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(microbenchmark)
library(imager)
# Source the C++ files
Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/ALS.cpp")
Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/CORE_ALS.cpp")
# Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/LEV_ALS.cpp")
# Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/PUNIF.cpp")
set.seed(42)
# 计算RMSE函数
compute_rmse <- function(data1, data2) {
  sqrt(sum((data1 - data2)^2, na.rm = TRUE)) / sqrt(sum(data1^2, na.rm = TRUE))
}

# 定义函数，处理单张图片并绘制结果
process_image <- function(image_path, lambda_, dim_factors, n_iter, sparsity, row_index) {
  # 读取图像并转换为灰度图
  img <- load.image(image_path)
  gray_img <- grayscale(img)
  
  # 转换为矩阵
  gray_matrix <- as.array(gray_img)
  N <- nrow(gray_matrix)
  M <- ncol(gray_matrix)
  
  # 稀疏化矩阵
  missing_indices <- matrix(runif(N * M) < sparsity, nrow = N, ncol = M)
  R_incomplete <- gray_matrix
  R_incomplete[missing_indices] <- NA
  mask <- !is.na(R_incomplete)
  R_incomplete <- matrix(R_incomplete, nrow = N, ncol = M)
  
  # 打印稀疏化矩阵
  img_corrupted <- as.cimg(R_incomplete)
  
  # 使用ALS还原
  result1 <- ALS(R_incomplete, lambda_, dim_factors, n_iter)
  R1 <- result1$U %*% result1$M
  rmse1 <- compute_rmse(R_incomplete[mask], R1[mask])
  
  # 使用CORE还原
  result2 <- FCORE(R_incomplete, lambda_, dim_factors, n_iter, 0.15)
  R2 <- result2$U %*% result2$M
  rmse2 <- compute_rmse(R_incomplete[mask], R2[mask])
  
  # 在指定的行中绘制图像
  plot(gray_img, main = paste("ORIGIN"), axes = FALSE)
  plot(img_corrupted, main = paste("CORRUPTED"), axes = FALSE)
  plot(as.cimg(R1), main = paste("ALS: RMSE = ", round(rmse1, 4)), axes = FALSE)
  plot(as.cimg(R2), main = paste("CORE: RMSE = ", round(rmse2, 4)), axes = FALSE)
}

# 主程序：循环显示3个不同图像的结果
par(mfrow = c(3, 4),  # 3行4列布局
    mar = c(2, 2, 2, 2),  # 缩小每行之间的间距，调整上下左右边距
    oma = c(0, 0, 2, 0))  # 设置外部边距

image_paths <- c("D:/R_language/R_code/new_ALS/figures/IMG1.jpg", 
                 "D:/R_language/R_code/new_ALS/figures/5530.jpg", 
                 "D:/R_language/R_code/new_ALS/figures/6378.jpg")

# 设置图像索引来确定每一行
for (i in 1:length(image_paths)) {
  # 每次调用 process_image 处理并绘制图片
  process_image(image_paths[i], lambda_ = 0.01, dim_factors = 100, n_iter = 5, sparsity = 0.6, row_index = i)
}
