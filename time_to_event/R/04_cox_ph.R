# Cox Proportional Hazards Models
# This script performs univariate and basic multivariate Cox PH analysis

# Load required libraries
library(survival)
library(forestplot)
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

# Function to perform univariate Cox analysis
perform_univariate_cox <- function(data_obj, dataset_name) {
  cat("\n=== Univariate Cox PH Analysis:", dataset_name, "===\n")
  
  data <- data_obj$data
  feature_names <- data_obj$feature_names
  
  # Prepare data for Cox models (remove missing values)
  # Use dplyr::select explicitly to avoid conflict with MASS::select
  data_clean <- data %>%
    dplyr::select(all_of(c("surv_obj", feature_names))) %>%
    na.omit()
  
  cat("Observations after removing missing values:", nrow(data_clean), "\n")
  
  # Univariate Cox models for each feature
  cox_results <- data.frame(
    feature = character(),
    coef = numeric(),
    hr = numeric(),
    hr_lower = numeric(),
    hr_upper = numeric(),
    se = numeric(),
    z = numeric(),
    pvalue = numeric(),
    stringsAsFactors = FALSE
  )
  
  cat("\n--- Fitting Univariate Cox Models ---\n")
  for (i in seq_along(feature_names)) {
    feature <- feature_names[i]
    
    if (i %% 10 == 0) {
      cat("Processing feature", i, "of", length(feature_names), ":", feature, "\n")
    }
    
    # Fit Cox model
    formula_str <- paste("surv_obj ~", feature)
    cox_fit <- tryCatch({
      coxph(as.formula(formula_str), data = data_clean)
    }, error = function(e) {
      cat("Error fitting", feature, ":", e$message, "\n")
      return(NULL)
    })
    
    if (is.null(cox_fit)) next
    
    # Extract results
    cox_summary <- summary(cox_fit)
    coef <- cox_summary$coefficients[1, "coef"]
    hr <- exp(coef)
    se <- cox_summary$coefficients[1, "se(coef)"]
    z <- cox_summary$coefficients[1, "z"]
    pvalue <- cox_summary$coefficients[1, "Pr(>|z|)"]
    ci_lower <- exp(coef - 1.96 * se)
    ci_upper <- exp(coef + 1.96 * se)
    
    cox_results <- rbind(cox_results, data.frame(
      feature = feature,
      coef = coef,
      hr = hr,
      hr_lower = ci_lower,
      hr_upper = ci_upper,
      se = se,
      z = z,
      pvalue = pvalue
    ))
  }
  
  # Sort by p-value
  cox_results <- cox_results[order(cox_results$pvalue), ]
  
  # Add significance indicator
  cox_results$significant <- ifelse(cox_results$pvalue < 0.05, "Yes", "No")
  cox_results$hr_formatted <- format_hr(cox_results$hr, 
                                         cox_results$hr_lower, 
                                         cox_results$hr_upper)
  
  cat("\n--- Top 20 Significant Features (p < 0.05) ---\n")
  significant_features <- cox_results[cox_results$pvalue < 0.05, ]
  print(head(significant_features, 20))
  
  # Save results
  save_table(cox_results, 
             paste0("cox_univariate_", dataset_name, ".csv"),
             "cox_results")
  
  # Forest plot for top features
  top_features <- head(cox_results, 30)  # Top 30 features
  
  # Create forest plot data
  forest_data <- data.frame(
    feature = top_features$feature,
    hr = top_features$hr,
    lower = top_features$hr_lower,
    upper = top_features$hr_upper,
    pvalue = top_features$pvalue
  )
  
  # Forest plot using ggplot2
  p_forest <- ggplot(forest_data, aes(x = hr, y = reorder(feature, hr))) +
    geom_point(size = 2, color = ifelse(forest_data$pvalue < 0.05, "red", "gray")) +
    geom_errorbarh(aes(xmin = lower, xmax = upper), 
                   height = 0.2, 
                   color = ifelse(forest_data$pvalue < 0.05, "red", "gray")) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
    labs(title = paste("Forest Plot: Top 30 Features (Univariate Cox)", dataset_name),
         x = "Hazard Ratio (95% CI)",
         y = "Feature") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8))
  
  save_figure(p_forest, 
              paste0("cox_models/forest_univariate_", dataset_name, ".png"),
              width = 12, height = 10)
  
  return(list(
    results = cox_results,
    significant_features = significant_features,
    n_features = length(feature_names),
    n_significant = sum(cox_results$pvalue < 0.05)
  ))
}

# Function to perform basic multivariate Cox analysis
perform_multivariate_cox <- function(data_obj, dataset_name, top_n = 20) {
  cat("\n=== Multivariate Cox PH Analysis:", dataset_name, "===\n")
  
  # First get univariate results
  univariate_results <- perform_univariate_cox(data_obj, dataset_name)
  
  # Select top features based on p-value
  top_features <- head(univariate_results$results, top_n)$feature
  
  cat("\n--- Fitting Multivariate Cox Model with Top", top_n, "Features ---\n")
  cat("Selected features:", paste(top_features, collapse = ", "), "\n")
  
  data <- data_obj$data
  
  # Prepare data
  data_clean <- data %>%
    dplyr::select(all_of(c("surv_obj", top_features))) %>%
    na.omit()
  
  cat("Observations after removing missing values:", nrow(data_clean), "\n")
  
  # Fit multivariate Cox model
  formula_str <- paste("surv_obj ~", paste(top_features, collapse = " + "))
  cox_multivariate <- coxph(as.formula(formula_str), data = data_clean)
  
  # Summary
  cox_summary <- summary(cox_multivariate)
  print(cox_summary)
  
  # Extract results
  cox_coef <- cox_summary$coefficients
  cox_results_multi <- data.frame(
    feature = rownames(cox_coef),
    coef = cox_coef[, "coef"],
    hr = exp(cox_coef[, "coef"]),
    hr_lower = exp(cox_coef[, "coef"] - 1.96 * cox_coef[, "se(coef)"]),
    hr_upper = exp(cox_coef[, "coef"] + 1.96 * cox_coef[, "se(coef)"]),
    se = cox_coef[, "se(coef)"],
    z = cox_coef[, "z"],
    pvalue = cox_coef[, "Pr(>|z|)"]
  )
  
  cox_results_multi$significant <- ifelse(cox_results_multi$pvalue < 0.05, "Yes", "No")
  cox_results_multi$hr_formatted <- format_hr(cox_results_multi$hr, 
                                               cox_results_multi$hr_lower, 
                                               cox_results_multi$hr_upper)
  
  # Sort by p-value
  cox_results_multi <- cox_results_multi[order(cox_results_multi$pvalue), ]
  
  cat("\n--- Multivariate Cox Model Results ---\n")
  print(cox_results_multi)
  
  # Model statistics
  cat("\n--- Model Statistics ---\n")
  cat("Concordance:", cox_summary$concordance["C"], "\n")
  cat("AIC:", AIC(cox_multivariate), "\n")
  cat("BIC:", BIC(cox_multivariate), "\n")
  
  # Save results
  save_table(cox_results_multi, 
             paste0("cox_multivariate_", dataset_name, ".csv"),
             "cox_results")
  
  # Forest plot for multivariate model
  p_forest_multi <- ggplot(cox_results_multi, aes(x = hr, y = reorder(feature, hr))) +
    geom_point(size = 2, color = ifelse(cox_results_multi$pvalue < 0.05, "red", "gray")) +
    geom_errorbarh(aes(xmin = hr_lower, xmax = hr_upper), 
                   height = 0.2, 
                   color = ifelse(cox_results_multi$pvalue < 0.05, "red", "gray")) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "black") +
    labs(title = paste("Forest Plot: Multivariate Cox Model", dataset_name),
         x = "Hazard Ratio (95% CI)",
         y = "Feature") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8))
  
  save_figure(p_forest_multi, 
              paste0("cox_models/forest_multivariate_", dataset_name, ".png"),
              width = 12, height = 8)
  
  return(list(
    model = cox_multivariate,
    results = cox_results_multi,
    summary = cox_summary,
    aic = AIC(cox_multivariate),
    bic = BIC(cox_multivariate),
    concordance = cox_summary$concordance["C"]
  ))
}

# Perform analysis for both datasets
cat("\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Cox Proportional Hazards Analysis\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

cox_uni_3year <- perform_univariate_cox(data_3year, "3year")
cox_uni_10year <- perform_univariate_cox(data_10year, "10year")

cox_multi_3year <- perform_multivariate_cox(data_3year, "3year", top_n = 20)
cox_multi_10year <- perform_multivariate_cox(data_10year, "10year", top_n = 20)

cat("\nCox PH analysis complete!\n")
cat("Results saved to results/reports/cox_results/\n")
cat("Figures saved to results/figures/cox_models/\n")
