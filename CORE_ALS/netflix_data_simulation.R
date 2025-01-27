# Load required libraries
library(Rcpp)
library(RcppArmadillo)
library(ggplot2)
library(microbenchmark)
# Source the C++ files
Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/ALS.cpp")
Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/CORE_ALS.cpp")
Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/FUNIF_ALS.cpp")
# Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/FLEV_ALS.cpp")
# Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/function/FIBOSS_ALS.cpp")
#Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/standard_function/UNIF_ALS.cpp")
Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/standard_function/IBOSS_ALS.cpp")
Rcpp::sourceCpp("D:/R_language/R_code/new_ALS/standard_function/LEV_ALS.cpp")
# RMSE computation function
compute_rmse <- function(data1, data2) {
  sqrt(sum((data1 - data2)^2, na.rm = TRUE)) / sqrt(sum(data1^2, na.rm = TRUE))
}


# 读取 CSV 文件并转化为矩阵
data <- read.csv("D:/R_language/R_code/new_ALS/40k_15k.csv", header = FALSE)  # 如果没有列名，可以设置 header = FALSE
R_incomplete <- as.matrix(data)[-1, -1]
# 删除所有为0的行
R_incomplete <- R_incomplete[apply(R_incomplete, 1, function(x) any(x != 0)), ]

# 删除所有为0的列
R_incomplete <- R_incomplete[, apply(R_incomplete, 2, function(x) any(x != 0))]

sparse_matrix <- R_incomplete[1:5000,1:5000]

non_zero_count <- sum(sparse_matrix != 0)

# 打印结果
print(paste("矩阵中非零元素的个数为:", non_zero_count))
# 将 0 替换为 NA
sparse_matrix[sparse_matrix == 0] <- NA


# 预设参数
iter_n <- 5  # 迭代次数
dim_factors <- c(20,25,30,35) # 隐向量维度
subsampling_rates <- c(0.1, 0.15, 0.2, 0.25)  # 抽样比例
nloop <- 5 # 循环次数
lambdas <- c(0.05,0.1,0.15)
# 结果矩阵初始化
rmse_fit_sub <- array(0, dim = c(5, length(lambdas), length(dim_factors), nloop, length(subsampling_rates)))  # 子样本拟合误差
rmse_pred_sub <- array(0, dim = c(5, length(lambdas), length(dim_factors), nloop, length(subsampling_rates)))  # 子样本预测误差


# 运行算法
for (t in 1:length(lambdas)) {
  cat("lambda:", lambdas[t], "\n")
  
  for (k in 1:length(dim_factors)) {
    cat("dim_factor =", dim_factors[k], "\n")
    
    
    # 运行基础ALS算法（R3）
    stime <- Sys.time()
    als <- ALS(sparse_matrix, lambdas[t], dim_factors[k], iter_n)  # 在这里使用UNIF方法
    cat("FULL ALS time (s):", as.numeric(Sys.time() - stime), "\n")
    U <- als$U
    M <- als$M
    R <- U %*% M
    rmse_fit_sub[1, t, k, , ] <- compute_rmse(sparse_matrix, R)
    rmse_pred_sub[1, t, k, , ] <- compute_rmse(sparse_matrix, R)
    for (i in 1:nloop) {
      cat("nloop", i, "\n")
      
      for (j in 1:length(subsampling_rates)) {
        cat("subsample_rate:", subsampling_rates[j], "\n")
        
        # 运行UNIF算法（以UNIF为例，其他方法类似）
        stime <- Sys.time()
        als0 <- unif(sparse_matrix, lambdas[t], dim_factors[k], iter_n, subsampling_rates[j])  # 在这里使用UNIF方法
        cat("UNIF ALS time (s):", as.numeric(Sys.time() - stime), "\n")
        U0 <- als0$U
        M0 <- als0$M
        R0 <- U0 %*% M0
        rmse_fit_sub[2, t, k, i, j] <- compute_rmse(sparse_matrix , R0)
        rmse_pred_sub[2, t, k, i, j] <- compute_rmse(sparse_matrix , R0)
        
        # 运行BLEV算法
        stime <- Sys.time()
        als1 <- lev(sparse_matrix, lambdas[t], dim_factors[k], iter_n, subsampling_rates[j])  # 在这里使用BLEV方法
        cat("BLEV ALS time (s):", as.numeric(Sys.time() - stime), "\n")
        U1 <- als1$U
        M1 <- als1$M
        R1 <- U1 %*% M1
        rmse_fit_sub[3, t, k, i, j] <- compute_rmse(sparse_matrix , R1)
        rmse_pred_sub[3, t, k, i, j] <- compute_rmse(sparse_matrix , R1)
        
        # 运行CORE算法
        stime <- Sys.time()
        als2 <- FCORE(sparse_matrix, lambdas[t], dim_factors[k], iter_n, subsampling_rates[j])  # 在这里使用CORE方法
        cat("CORE ALS time (s):", as.numeric(Sys.time() - stime), "\n")
        U2 <- als2$U
        M2 <- als2$M
        R2 <- U2 %*% M2
        rmse_fit_sub[4, t, k, i, j] <- compute_rmse(sparse_matrix , R2)
        rmse_pred_sub[4, t, k, i, j] <- compute_rmse(sparse_matrix , R2)
        
        # 运行IBOSS算法
        stime <- Sys.time()
        als3 <- IBOSS(sparse_matrix, lambdas[t], dim_factors[k], iter_n,   subsampling_rates[j])  # 在这里使用CORE方法
        cat("IBOSS ALS time (s):", as.numeric(Sys.time() - stime), "\n")
        U3 <- als3$U
        M3 <- als3$M
        R3 <- U3 %*% M3
        rmse_fit_sub[5, t, k, i, j] <- compute_rmse(sparse_matrix , R3)
        rmse_pred_sub[5, t, k, i, j] <- compute_rmse(sparse_matrix , R3)
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
    #method_index <- match(method, c("ALS","UNIF", "BLEV", "CORE"))
    method_index <- match(method, c("ALS","UNIF", "BLEV", "CORE","IBOSS"))
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
  Method = c("ALS","UNIF", "BLEV", "CORE"),
  #Method = c("ALS","UNIF", "BLEV", "CORE","IBOSS"),
  R_list = c("L1", "L2", "L3"), 
  M_list = c("I1", "I2", "I3", "I4"),
  rmse_fit_sub = rmse_fit_sub
)


ggplot(df, aes(x = subsample, y = MSE, color = Method, group = Method, linetype = Method)) +
  geom_line(size = 1) +  # 增加折线的粗细
  geom_point(size = 2) +  # 添加点，设置点的大小
  geom_errorbar(aes(ymin = MSE - MSE_error, ymax = MSE + MSE_error), 
                width = 50, size = 1,linetype = "solid") +  # 误差条粗细
  facet_wrap(~subplot_label, nrow = 3, ncol = 4, scales = "free_y", labeller = label_value) +  # 设置子图布局
  labs(x = "Subsampling Rate", y = "RMSE") +  # 设置轴标签
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
      "CORE" = "red",      # CORE 的颜色设置为红色
      "IBOSS" = "pink"
    )
  )+ 
  scale_linetype_manual(
    values = c(
      "ALS" = "dotdash",    # ALS 使用长虚线
      "UNIF" = "longdash",     # UNIF 使用实线
      "BLEV" = "dashed",    # BLEV 使用虚线
      "CORE" = "solid",    # CORE 使用点线"
      "IBOSS" = "twodash"
    )
  )

