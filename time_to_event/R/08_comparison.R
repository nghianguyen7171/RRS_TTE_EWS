# Dataset Comparison
# This script compares 3-year and 10-year datasets

# Load required libraries
library(survival)
library(survminer)
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

# Function to compare basic statistics
compare_basic_stats <- function(data_3year, data_10year) {
  cat("\n=== Basic Statistics Comparison ===\n")
  
  stats_3year <- data_3year$data %>%
    summarise(
      dataset = "3year",
      n_obs = n(),
      n_patients = length(unique(patient_id)),
      n_events = sum(event_occurred),
      event_rate = mean(event_occurred),
      median_time = median(time_to_event, na.rm = TRUE),
      mean_time = mean(time_to_event, na.rm = TRUE),
      n_features = length(data_3year$feature_names)
    )
  
  stats_10year <- data_10year$data %>%
    summarise(
      dataset = "10year",
      n_obs = n(),
      n_patients = length(unique(patient_id)),
      n_events = sum(event_occurred),
      event_rate = mean(event_occurred),
      median_time = median(time_to_event, na.rm = TRUE),
      mean_time = mean(time_to_event, na.rm = TRUE),
      n_features = length(data_10year$feature_names)
    )
  
  comparison_stats <- rbind(stats_3year, stats_10year)
  print(comparison_stats)
  
  save_table(comparison_stats, "dataset_comparison_basic_stats.csv", "model_summaries")
  
  return(comparison_stats)
}

# Function to compare survival curves
compare_survival_curves <- function(data_3year, data_10year) {
  cat("\n=== Survival Curves Comparison ===\n")
  
  # Combine datasets - ensure column names match
  data_3year_df <- data_3year$data %>% mutate(dataset = "3year")
  data_10year_df <- data_10year$data %>% mutate(dataset = "10year")
  
  # Get common columns (ensure surv_obj, time_to_event, event_occurred are included)
  common_cols <- intersect(colnames(data_3year_df), colnames(data_10year_df))
  required_cols <- c("time_to_event", "event_occurred", "dataset")
  common_cols <- unique(c(required_cols, common_cols))
  common_cols <- common_cols[common_cols %in% colnames(data_3year_df) & 
                              common_cols %in% colnames(data_10year_df)]
  
  data_combined <- rbind(
    data_3year_df %>% dplyr::select(all_of(common_cols)),
    data_10year_df %>% dplyr::select(all_of(common_cols))
  )
  
  # Recreate survival object
  library(survival)
  data_combined$surv_obj <- Surv(
    time = data_combined$time_to_event,
    event = data_combined$event_occurred
  )
  
  # Fit Kaplan-Meier for each dataset
  km_fit <- survfit(surv_obj ~ dataset, data = data_combined)
  
  # Log-rank test
  logrank_test <- survdiff(surv_obj ~ dataset, data = data_combined)
  pval <- 1 - pchisq(logrank_test$chisq, length(logrank_test$n) - 1)
  
  cat("Log-rank test p-value:", format_pvalue(pval), "\n")
  
  # Plot comparison
  p_comparison <- ggsurvplot(
    km_fit,
    data = data_combined,
    title = "Survival Curves: 3-year vs 10-year Datasets",
    xlab = "Time (hours)",
    ylab = "Survival Probability",
    risk.table = TRUE,
    conf.int = TRUE,
    pval = TRUE,
    pval.method = TRUE,
    surv.median.line = "hv",
    legend.title = "Dataset",
    legend.labs = c("3-year", "10-year"),
    ggtheme = theme_minimal()
  )
  
  save_figure(p_comparison$plot, 
              "comparisons/survival_curves_comparison.png",
              width = 10, height = 8)
  
  return(list(
    km_fit = km_fit,
    logrank_pvalue = pval
  ))
}

# Function to compare Cox model results
compare_cox_models <- function(data_3year, data_10year) {
  cat("\n=== Cox Model Results Comparison ===\n")
  
  # Load Cox results
  if (!exists("perform_multivariate_cox")) {
    source("R/04_cox_ph.R", local = TRUE)
  }
  cox_3year <- perform_multivariate_cox(data_3year, "3year", top_n = 20)
  cox_10year <- perform_multivariate_cox(data_10year, "10year", top_n = 20)
  
  # Compare model performance
  performance_comparison <- data.frame(
    dataset = c("3year", "10year"),
    concordance = c(cox_3year$concordance, cox_10year$concordance),
    aic = c(cox_3year$aic, cox_10year$aic),
    bic = c(cox_3year$bic, cox_10year$bic),
    n_features = c(nrow(cox_3year$results), nrow(cox_10year$results))
  )
  
  cat("\n--- Model Performance Comparison ---\n")
  print(performance_comparison)
  
  # Compare significant features
  sig_3year <- cox_3year$results[cox_3year$results$pvalue < 0.05, "feature"]
  sig_10year <- cox_10year$results[cox_10year$results$pvalue < 0.05, "feature"]
  
  common_features <- intersect(sig_3year, sig_10year)
  unique_3year <- setdiff(sig_3year, sig_10year)
  unique_10year <- setdiff(sig_10year, sig_3year)
  
  cat("\n--- Significant Features Comparison ---\n")
  cat("Common significant features:", length(common_features), "\n")
  if (length(common_features) > 0) {
    cat(paste(common_features, collapse = ", "), "\n")
  }
  
  cat("\nUnique to 3-year:", length(unique_3year), "\n")
  if (length(unique_3year) > 0 && length(unique_3year) <= 20) {
    cat(paste(unique_3year, collapse = ", "), "\n")
  }
  
  cat("\nUnique to 10-year:", length(unique_10year), "\n")
  if (length(unique_10year) > 0 && length(unique_10year) <= 20) {
    cat(paste(unique_10year, collapse = ", "), "\n")
  }
  
  # Compare hazard ratios for common features
  if (length(common_features) > 0) {
    hr_comparison <- merge(
      cox_3year$results[cox_3year$results$feature %in% common_features, 
                       c("feature", "hr", "hr_lower", "hr_upper", "pvalue")],
      cox_10year$results[cox_10year$results$feature %in% common_features, 
                        c("feature", "hr", "hr_lower", "hr_upper", "pvalue")],
      by = "feature",
      suffixes = c("_3year", "_10year")
    )
    
    cat("\n--- Hazard Ratio Comparison (Common Features) ---\n")
    print(head(hr_comparison, 10))
    
    save_table(hr_comparison, "hr_comparison_common_features.csv", "model_summaries")
    
    # Visualization: HR comparison
    p_hr_comparison <- ggplot(hr_comparison, aes(x = hr_3year, y = hr_10year)) +
      geom_point(alpha = 0.6, size = 2) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
      geom_errorbarh(aes(xmin = hr_lower_3year, xmax = hr_upper_3year), alpha = 0.3) +
      geom_errorbar(aes(ymin = hr_lower_10year, ymax = hr_upper_10year), alpha = 0.3) +
      labs(title = "Hazard Ratio Comparison: 3-year vs 10-year",
           x = "Hazard Ratio (3-year)",
           y = "Hazard Ratio (10-year)") +
      theme_minimal()
    
    save_figure(p_hr_comparison, 
                "comparisons/hr_comparison.png",
                width = 10, height = 8)
  }
  
  save_table(performance_comparison, "model_performance_comparison.csv", "model_summaries")
  
  return(list(
    performance = performance_comparison,
    common_features = common_features,
    unique_3year = unique_3year,
    unique_10year = unique_10year
  ))
}

# Function to compare feature importance
compare_feature_importance <- function(data_3year, data_10year) {
  cat("\n=== Feature Importance Comparison ===\n")
  
  # Load feature importance
  if (!exists("calculate_feature_importance")) {
    source("R/06_feature_importance.R", local = TRUE)
  }
  importance_3year <- calculate_feature_importance(data_3year, "3year", model_type = "multivariate")
  importance_10year <- calculate_feature_importance(data_10year, "10year", model_type = "multivariate")
  
  # Get top 20 features from each
  top_3year <- head(importance_3year, 20)
  top_10year <- head(importance_10year, 20)
  
  # Compare rankings
  common_top <- intersect(top_3year$feature, top_10year$feature)
  
  cat("Common features in top 20:", length(common_top), "\n")
  if (length(common_top) > 0) {
    cat(paste(common_top, collapse = ", "), "\n")
  }
  
  # Create comparison table
  comparison_importance <- merge(
    top_3year[, c("feature", "importance_score")],
    top_10year[, c("feature", "importance_score")],
    by = "feature",
    suffixes = c("_3year", "_10year"),
    all = TRUE
  )
  
  comparison_importance <- comparison_importance %>%
    arrange(desc(pmax(importance_score_3year, importance_score_10year, na.rm = TRUE)))
  
  save_table(comparison_importance, "feature_importance_comparison.csv", "model_summaries")
  
  # Visualization
  if (length(common_top) > 0) {
    comparison_common <- comparison_importance[comparison_importance$feature %in% common_top, ]
    
    p_importance_comparison <- ggplot(comparison_common, 
                                      aes(x = importance_score_3year, 
                                          y = importance_score_10year)) +
      geom_point(size = 3, alpha = 0.6) +
      geom_text(aes(label = feature), hjust = 0, vjust = 0, size = 3) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
      labs(title = "Feature Importance Comparison: 3-year vs 10-year",
           x = "Importance Score (3-year)",
           y = "Importance Score (10-year)") +
      theme_minimal()
    
    save_figure(p_importance_comparison, 
                "comparisons/feature_importance_comparison.png",
                width = 12, height = 10)
  }
  
  return(comparison_importance)
}

# Perform all comparisons
cat("\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Dataset Comparison: 3-year vs 10-year\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

basic_stats <- compare_basic_stats(data_3year, data_10year)
survival_comparison <- compare_survival_curves(data_3year, data_10year)
cox_comparison <- compare_cox_models(data_3year, data_10year)
importance_comparison <- compare_feature_importance(data_3year, data_10year)

cat("\nDataset comparison complete!\n")
cat("Results saved to results/reports/model_summaries/\n")
cat("Figures saved to results/figures/comparisons/\n")
