# Model Diagnostics
# This script performs diagnostic tests for Cox models

# Load required libraries
library(survival)
library(ggplot2)
library(dplyr)

# Source utility functions
source("R/utils.R")

# Set project directory
project_root <- get_project_root()
setwd(project_root)

# Load data (assumes 01_load_data.R has been run)
if (!exists("data_3year") || !exists("data_10year")) {
  source("R/01_load_data.R")
}

# Function to test proportional hazards assumption
test_proportional_hazards <- function(data_obj, dataset_name, top_n = 10) {
  cat("\n=== Proportional Hazards Assumption Test:", dataset_name, "===\n")
  
  # Get top features
  if (!exists("calculate_feature_importance")) {
    source("R/06_feature_importance.R", local = TRUE)
  }
  importance_df <- calculate_feature_importance(data_obj, dataset_name, model_type = "multivariate")
  top_features <- head(importance_df, top_n)$feature
  
  data <- data_obj$data
  
  # Prepare data
  data_clean <- data %>%
    select(all_of(c("surv_obj", top_features))) %>%
    na.omit()
  
  # Fit Cox model
  formula_str <- paste("surv_obj ~", paste(top_features, collapse = " + "))
  cox_model <- coxph(as.formula(formula_str), data = data_clean)
  
  # Test proportional hazards using Schoenfeld residuals
  ph_test <- cox.zph(cox_model)
  
  cat("\n--- Proportional Hazards Test Results ---\n")
  print(ph_test)
  
  # Global test
  global_pval <- ph_test$table["GLOBAL", "p"]
  cat("\nGlobal test p-value:", format_pvalue(global_pval), "\n")
  
  if (global_pval < 0.05) {
    cat("WARNING: Proportional hazards assumption may be violated (p < 0.05)\n")
  } else {
    cat("Proportional hazards assumption appears valid (p >= 0.05)\n")
  }
  
  # Plot Schoenfeld residuals
  png(file.path(get_results_dir(), "figures", "diagnostics", 
                paste0("schoenfeld_residuals_", dataset_name, ".png")),
      width = 12, height = 8, units = "in", res = 300)
  par(mfrow = c(2, ceiling(length(top_features) / 2)))
  plot(ph_test)
  dev.off()
  
  # Save test results
  ph_results_df <- as.data.frame(ph_test$table)
  ph_results_df$feature <- rownames(ph_results_df)
  ph_results_df <- ph_results_df[, c("feature", "chisq", "df", "p")]
  
  save_table(ph_results_df, 
             paste0("ph_assumption_test_", dataset_name, ".csv"),
             "model_summaries")
  
  return(list(
    ph_test = ph_test,
    global_pvalue = global_pval,
    results = ph_results_df
  ))
}

# Function to analyze residuals
analyze_residuals <- function(data_obj, dataset_name, top_n = 10) {
  cat("\n=== Residual Analysis:", dataset_name, "===\n")
  
  # Get top features
  if (!exists("calculate_feature_importance")) {
    source("R/06_feature_importance.R", local = TRUE)
  }
  importance_df <- calculate_feature_importance(data_obj, dataset_name, model_type = "multivariate")
  top_features <- head(importance_df, top_n)$feature
  
  data <- data_obj$data
  
  # Prepare data
  data_clean <- data %>%
    select(all_of(c("surv_obj", top_features))) %>%
    na.omit()
  
  # Fit Cox model
  formula_str <- paste("surv_obj ~", paste(top_features, collapse = " + "))
  cox_model <- coxph(as.formula(formula_str), data = data_clean)
  
  # Calculate residuals
  martingale_residuals <- residuals(cox_model, type = "martingale")
  deviance_residuals <- residuals(cox_model, type = "deviance")
  dfbeta_residuals <- residuals(cox_model, type = "dfbeta")
  
  # Combine residuals
  residuals_df <- data.frame(
    martingale = martingale_residuals,
    deviance = deviance_residuals,
    time = data_clean$surv_obj[, "time"],
    event = data_clean$surv_obj[, "status"]
  )
  
  # Summary statistics
  cat("\n--- Residual Summary Statistics ---\n")
  cat("Martingale residuals:\n")
  print(summary(martingale_residuals))
  cat("\nDeviance residuals:\n")
  print(summary(deviance_residuals))
  
  # Visualizations
  
  # Martingale residuals vs time
  p_martingale <- ggplot(residuals_df, aes(x = time, y = martingale)) +
    geom_point(alpha = 0.5, color = ifelse(residuals_df$event == 1, "red", "blue")) +
    geom_smooth(method = "loess", se = TRUE) +
    labs(title = paste("Martingale Residuals vs Time:", dataset_name),
         x = "Time",
         y = "Martingale Residuals") +
    theme_minimal()
  
  save_figure(p_martingale, 
              paste0("diagnostics/martingale_residuals_", dataset_name, ".png"),
              width = 10, height = 6)
  
  # Deviance residuals vs time
  p_deviance <- ggplot(residuals_df, aes(x = time, y = deviance)) +
    geom_point(alpha = 0.5, color = ifelse(residuals_df$event == 1, "red", "blue")) +
    geom_smooth(method = "loess", se = TRUE) +
    labs(title = paste("Deviance Residuals vs Time:", dataset_name),
         x = "Time",
         y = "Deviance Residuals") +
    theme_minimal()
  
  save_figure(p_deviance, 
              paste0("diagnostics/deviance_residuals_", dataset_name, ".png"),
              width = 10, height = 6)
  
  # Residual distributions
  p_resid_dist <- ggplot(residuals_df, aes(x = martingale)) +
    geom_histogram(bins = 50, fill = "steelblue", alpha = 0.7) +
    labs(title = paste("Distribution of Martingale Residuals:", dataset_name),
         x = "Martingale Residuals",
         y = "Frequency") +
    theme_minimal()
  
  save_figure(p_resid_dist, 
              paste0("diagnostics/residual_distribution_", dataset_name, ".png"),
              width = 10, height = 6)
  
  # Save residuals
  save_table(residuals_df, 
             paste0("residuals_", dataset_name, ".csv"),
             "model_summaries")
  
  return(residuals_df)
}

# Function to assess model fit
assess_model_fit <- function(data_obj, dataset_name, top_n = 10) {
  cat("\n=== Model Fit Assessment:", dataset_name, "===\n")
  
  # Get top features
  if (!exists("calculate_feature_importance")) {
    source("R/06_feature_importance.R", local = TRUE)
  }
  importance_df <- calculate_feature_importance(data_obj, dataset_name, model_type = "multivariate")
  top_features <- head(importance_df, top_n)$feature
  
  data <- data_obj$data
  
  # Prepare data
  data_clean <- data %>%
    select(all_of(c("surv_obj", top_features))) %>%
    na.omit()
  
  # Fit Cox model
  formula_str <- paste("surv_obj ~", paste(top_features, collapse = " + "))
  cox_model <- coxph(as.formula(formula_str), data = data_clean)
  
  # Model summary
  model_summary <- summary(cox_model)
  
  # Concordance (C-index)
  concordance <- model_summary$concordance["C"]
  cat("\n--- Model Performance Metrics ---\n")
  cat("Concordance (C-index):", concordance, "\n")
  cat("AIC:", AIC(cox_model), "\n")
  cat("BIC:", BIC(cox_model), "\n")
  
  # Likelihood ratio test
  # Compare with null model
  cox_null <- coxph(surv_obj ~ 1, data = data_clean)
  lrt <- anova(cox_null, cox_model, test = "LRT")
  
  cat("\n--- Likelihood Ratio Test ---\n")
  print(lrt)
  
  # Save fit metrics
  fit_metrics <- data.frame(
    metric = c("Concordance", "AIC", "BIC", "LRT_pvalue"),
    value = c(concordance, AIC(cox_model), BIC(cox_model), 
              lrt$`Pr(>Chi)`[2])
  )
  
  save_table(fit_metrics, 
             paste0("model_fit_metrics_", dataset_name, ".csv"),
             "model_summaries")
  
  return(list(
    concordance = concordance,
    aic = AIC(cox_model),
    bic = BIC(cox_model),
    lrt = lrt,
    metrics = fit_metrics
  ))
}

# Perform diagnostics for both datasets
cat("\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Model Diagnostics\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

ph_test_3year <- test_proportional_hazards(data_3year, "3year", top_n = 10)
ph_test_10year <- test_proportional_hazards(data_10year, "10year", top_n = 10)

residuals_3year <- analyze_residuals(data_3year, "3year", top_n = 10)
residuals_10year <- analyze_residuals(data_10year, "10year", top_n = 10)

fit_3year <- assess_model_fit(data_3year, "3year", top_n = 10)
fit_10year <- assess_model_fit(data_10year, "10year", top_n = 10)

cat("\nModel diagnostics complete!\n")
cat("Results saved to results/reports/model_summaries/\n")
cat("Figures saved to results/figures/diagnostics/\n")
