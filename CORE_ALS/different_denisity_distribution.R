library(Matrix)
library(ggplot2)
library(Rcpp)
library(RcppArmadillo)
library(microbenchmark)
library(dplyr)
library(scales)
# Source C++ files
Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/ALS.cpp")
Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/FCORE_ALS.cpp")
Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/FLEV_ALS.cpp")
Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/FUNIF_ALS.cpp")
Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/FIBOSS_ALS.cpp")


# RMSE computation function
compute_rmse <- function(data1, data2) {
  sqrt(sum((data1 - data2)^2, na.rm = TRUE)) / sqrt(sum(data1^2, na.rm = TRUE))
}

# 预设参数
iter_n <- 5  # 迭代次数
lambda0 <- 0.2
dim_factors <- 60  # 隐向量维度
num_rows <- 3600
num_cols <- 3600
denisities <- c(0.4, 0.5, 0.6, 0.7)  # 稀疏度
subsampling_rates <- c(0.1, 0.15, 0.2, 0.25)  # 抽样比例
nloop <- 10 # 循环次数
distributions <- c("D1", "D2", "D3")

# 结果矩阵初始化
rmse_fit_sub <- array(0, dim = c(5, length(distributions), length(denisities), nloop, length(subsampling_rates)))  # 子样本拟合误差
rmse_pred_sub <- array(0, dim = c(5, length(distributions), length(denisities), nloop, length(subsampling_rates)))  # 子样本预测误差

# 定义矩阵生成函数
generate_distribution_matrix <- function(dim, n_samples, choice) {
  # Sigma and covariance matrix
  sigma <- 0.6
  cov_matrix <- outer(1:dim, 1:dim, function(i, j) sigma^abs(i - j))
  
  if (choice == 1) {
    # D1: Multivariate normal distribution
    return(MASS::mvrnorm(n_samples, mu = rep(0, dim), Sigma = cov_matrix))
  } else if (choice == 2) {
    # D2: Multivariate log-normal distribution
    X_lognormal <- MASS::mvrnorm(n_samples, mu = rep(0, dim), Sigma = cov_matrix)
    return(exp(X_lognormal))
  }else if (choice == 3) {
    # D3: Multivariate t-distribution with 5 degrees of freedom
    df <- 5
    X_t_dist <- mvtnorm::rmvt(n_samples, delta = rep(0, dim), sigma = cov_matrix, df = df)
    return(X_t_dist)
  } else {
    stop("Choice must be 1, 2, or 3.")
  }
}


# 运行算法
for (t in 1:length(distributions)) {
  cat("distribution:", distributions[t], "\n")
  
  for (k in 1:length(denisities)) {
    cat("denisity =", denisities[k], "\n")
    
    # 生成随机稀疏矩阵
    U_true <- generate_distribution_matrix(dim_factors, num_rows, t)
    V_true <- t(generate_distribution_matrix(dim_factors, num_cols, t))
    R_true <- U_true %*% V_true
    
    num_nan <- round(num_rows * num_cols * (1 - denisities[k]))  # NaN数量
    nan_indices <- sample(num_rows * num_cols, num_nan)  # 随机选择NaN的位置
    sparse_matrix <- R_true
    sparse_matrix[as.vector(nan_indices)] <- NA  # 设置NaN值
    pred_matrix <- R_true
    pred_matrix[as.vector(!nan_indices)] <- NA 
    
    # 运行基础ALS算法（R3）
    stime <- Sys.time()
    als <- ALS(sparse_matrix, lambda0, dim_factors, iter_n)  # 在这里使用UNIF方法
    cat("FULL ALS time (s):", as.numeric(Sys.time() - stime), "\n")
    U <- als$U
    M <- als$M
    R <- U %*% M
    rmse_fit_sub[1, t, k, , ] <- compute_rmse(sparse_matrix, R)
    rmse_pred_sub[1, t, k, , ] <- compute_rmse(pred_matrix, R)
    for (i in 1:nloop) {
      cat("nloop", i, "\n")
      
      for (j in 1:length(subsampling_rates)) {
        cat("subsample_rate:", subsampling_rates[j], "\n")
        
        # 运行UNIF算法（以UNIF为例，其他方法类似）
        stime <- Sys.time()
        als0 <- unif(sparse_matrix, lambda0, dim_factors, iter_n, subsampling_rates[j])  # 在这里使用UNIF方法
        cat("UNIF ALS time (s):", as.numeric(Sys.time() - stime), "\n")
        U0 <- als0$U
        M0 <- als0$M
        R0 <- U0 %*% M0
        rmse_fit_sub[2, t, k, i, j] <- compute_rmse(sparse_matrix, R0)
        rmse_pred_sub[2, t, k, i, j] <- compute_rmse(pred_matrix, R0)
        
        # 运行BLEV算法
        stime <- Sys.time()
        als1 <- lev(sparse_matrix, lambda0, dim_factors, iter_n, subsampling_rates[j])  # 在这里使用BLEV方法
        cat("BLEV ALS time (s):", as.numeric(Sys.time() - stime), "\n")
        U1 <- als1$U
        M1 <- als1$M
        R1 <- U1 %*% M1
        rmse_fit_sub[3, t, k, i, j] <- compute_rmse(sparse_matrix, R1)
        rmse_pred_sub[3, t, k, i, j] <- compute_rmse(pred_matrix, R1)
        
        # 运行CORE算法
        stime <- Sys.time()
        als2 <- FCORE(sparse_matrix, lambda0, dim_factors, iter_n, subsampling_rates[j])  # 在这里使用CORE方法
        cat("CORE ALS time (s):", as.numeric(Sys.time() - stime), "\n")
        U2 <- als2$U
        M2 <- als2$M
        R2 <- U2 %*% M2
        rmse_fit_sub[4, t, k, i, j] <- compute_rmse(sparse_matrix, R2)
        rmse_pred_sub[4, t, k, i, j] <- compute_rmse(pred_matrix, R2)
        
        # # 运行IBOSS算法
        # stime <- Sys.time()
        # als3 <- IBOSS(sparse_matrix, lambda0, dim_factors, iter_n,   subsampling_rates[j])  # 在这里使用CORE方法
        # cat("IBOSS ALS time (s):", as.numeric(Sys.time() - stime), "\n")
        # U3 <- als3$U
        # M3 <- als3$M
        # R3 <- U3 %*% M3
        # rmse_fit_sub[5, t, k, i, j] <- compute_rmse(sparse_matrix, R3)
        # rmse_pred_sub[5, t, k, i, j] <- compute_rmse(pred_matrix, R3)
      }
    }
  }
}











df_init <- function(subsample_sizes = subsampling_rates,
                    Method = c("ALS","UNIF", "BLEV", "CORE","IBOSS"),
                    R_list = c("D1", "D2", "D3"), 
                    M_list = c("R1", "R2", "R3", "R4"),
                    rmse_fit_sub) {
  
  set.seed(123)
  
  # 使用 expand.grid() 创建所有可能的组合
  data <- expand.grid(
    Method = Method,
    subsample = subsample_sizes,
    R = R_list,
    M = M_list
  )
  
  # 初始化 MSE 和误差条列
  data$MSE <- NA
  data$MSE_error <- NA
  
  # 遍历每个组合并计算 MSE 和误差条
  for (i in 1:nrow(data)) {
    method <- data$Method[i]
    subsample <- data$subsample[i]
    R <- data$R[i]
    M <- data$M[i]
    
    # 查找对应的索引
    method_index <- match(method, c("ALS","UNIF", "BLEV", "FCORE"))
    dist_index <- match(R, R_list)
    dens_index <- match(M, M_list)
    subsample_index <- match(subsample, subsampling_rates)
    
    # 从 rmse_fit_sub 中提取所有循环的 RMSE 数据
    rmse_values <- rmse_fit_sub[method_index, dist_index, dens_index, , subsample_index]
    
    # 计算均值和误差条（最大值与最小值的差）
    data$MSE[i] <- mean(rmse_values)  # 均值作为 MSE
    data$MSE_error[i] <- (max(rmse_values) - min(rmse_values))  # 最大值与最小值的差作为误差条
  }
  
  # 创建子图标签
  data$subplot_label <- paste(data$R, data$M, sep = ",")
  
  return(data)
}



# 调用 df_init 并传入 rmse_fit_sub
df <- df_init(
  subsample_sizes = subsampling_rates,
  Method = c("ALS","UNIF", "BLEV", "FCORE"),
  R_list = c("D1", "D2", "D3"), 
  M_list = c("R1", "R2", "R3", "R4"),
  rmse_fit_sub = rmse_pred_sub
)


ggplot(df, aes(x = subsample, y = MSE, color = Method, group = Method, linetype = Method)) +
  geom_line(size = 1) +  # 增加折线的粗细
  geom_point(size = 2) +  # 添加点，设置点的大小
  geom_errorbar(aes(ymin = MSE - MSE_error, ymax = MSE + MSE_error), 
                width = 50, size = 1,linetype = "solid") +  # 误差条粗细
  facet_wrap(~subplot_label, nrow = 3, ncol = 4, scales = "free_y", labeller = label_value) +  # 设置子图布局
  labs(x = "Subsampling Rate", y = "PRMSE") +  # 设置轴标签
  scale_x_continuous(
    breaks = subsampling_rates,  # 设置四个刻度
    limits = c(subsampling_rates[1], subsampling_rates[4])  # 设置横坐标的范围
  ) + 
  theme_minimal() +  # 使用简洁主题
  theme(
    legend.position = "bottom",  # 图例放到底部
    legend.text = element_text(size = 12),  # 图例文本大小
    legend.title = element_text(size = 14),  # 图例标题文本大小
    strip.text = element_text(size = 12, face = "bold"),  # 子图标题加粗
    panel.border = element_rect(color = "black", fill = NA, linewidth = 1),  # 添加边框
    axis.title = element_text(size = 14, face = "bold"),  # 坐标轴标题加粗
    axis.text = element_text(size = 10),  # 坐标轴刻度标签字体大小
    axis.ticks = element_line(color = "black", linewidth = 0.5),  # 坐标轴刻度线
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5),  # 图形标题加粗居中
    plot.margin = margin(10, 10, 10, 10)  # 设置图形边距，避免裁剪
  ) +
  scale_color_manual(
    values = c(
      "ALS" = "#808080" ,     # ALS 的颜色设置为黑色
      "UNIF" = "blue",     # UNIF 的颜色设置为蓝色4e
      "BLEV" = "#8B008B",   # BLEV 的颜色设置为紫色
      "FCORE" = "red",      # CORE 的颜色设置为红色
      "IBOSS" = "pink"
    )
  )+ 
  scale_linetype_manual(
    values = c(
      "ALS" = "dotdash",    # ALS 使用长虚线
      "UNIF" = "longdash",     # UNIF 使用实线
      "BLEV" = "dashed",    # BLEV 使用虚线
      "FCORE" = "solid",    # CORE 使用点线"
      "IBOSS" = "twodash"
    )
  )