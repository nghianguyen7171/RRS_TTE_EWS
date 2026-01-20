# Feature Importance Analysis
# This script analyzes feature importance and develops risk scores

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

# Function to calculate feature importance from Cox models
calculate_feature_importance <- function(data_obj, dataset_name, model_type = "multivariate") {
  cat("\n=== Feature Importance Analysis:", dataset_name, "===\n")
  
  # Load Cox model results
  if (model_type == "multivariate") {
    # Use results from multivariate analysis
    if (!exists("perform_multivariate_cox")) {
      source("R/04_cox_ph.R", local = TRUE)
    }
    cox_results <- perform_multivariate_cox(data_obj, dataset_name, top_n = 20)
    results_df <- cox_results$results
  } else {
    # Use univariate results
    if (!exists("perform_univariate_cox")) {
      source("R/04_cox_ph.R", local = TRUE)
    }
    cox_results <- perform_univariate_cox(data_obj, dataset_name)
    results_df <- cox_results$results
  }
  
  # Calculate importance metrics
  importance_df <- results_df %>%
    mutate(
      # Absolute coefficient (magnitude of effect)
      abs_coef = abs(coef),
      # Absolute z-score (statistical significance)
      abs_z = abs(z),
      # -log10(p-value) (higher = more significant)
      neg_log10_p = -log10(pvalue),
      # Combined importance score (weighted combination)
      importance_score = abs_coef * neg_log10_p
    ) %>%
    arrange(desc(importance_score))
  
  cat("\n--- Top 20 Most Important Features ---\n")
  print(head(importance_df, 20))
  
  # Save importance rankings
  save_table(importance_df, 
             paste0("feature_importance_", model_type, "_", dataset_name, ".csv"),
             "model_summaries")
  
  # Visualization: Feature importance plot
  top_20 <- head(importance_df, 20)
  
  p_importance <- ggplot(top_20, aes(x = importance_score, y = reorder(feature, importance_score))) +
    geom_col(fill = "steelblue", alpha = 0.7) +
    labs(title = paste("Feature Importance (Top 20):", dataset_name),
         x = "Importance Score",
         y = "Feature") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8))
  
  save_figure(p_importance, 
              paste0("cox_models/feature_importance_", model_type, "_", dataset_name, ".png"),
              width = 10, height = 8)
  
  # Visualization: Coefficient vs p-value
  p_coef_pval <- ggplot(results_df, aes(x = coef, y = -log10(pvalue))) +
    geom_point(aes(color = ifelse(pvalue < 0.05, "Significant", "Not Significant")), 
               size = 2, alpha = 0.6) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", color = "red") +
    geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
    scale_color_manual(values = c("Significant" = "red", "Not Significant" = "gray")) +
    labs(title = paste("Coefficient vs P-value:", dataset_name),
         x = "Coefficient",
         y = "-log10(p-value)",
         color = "Significance") +
    theme_minimal()
  
  save_figure(p_coef_pval, 
              paste0("cox_models/coef_vs_pval_", dataset_name, ".png"),
              width = 10, height = 8)
  
  return(importance_df)
}

# Function to develop risk score
develop_risk_score <- function(data_obj, dataset_name, top_n = 10) {
  cat("\n=== Risk Score Development:", dataset_name, "===\n")
  
  # Get top features from importance analysis
  importance_df <- calculate_feature_importance(data_obj, dataset_name, model_type = "multivariate")
  top_features <- head(importance_df, top_n)$feature
  
  cat("Using top", top_n, "features for risk score:\n")
  cat(paste(top_features, collapse = ", "), "\n")
  
  data <- data_obj$data
  
  # Prepare data - include event_occurred and time_to_event
  data_clean <- data %>%
    select(all_of(c("surv_obj", "patient_id", "time_to_event", "event_occurred", top_features))) %>%
    na.omit()
  
  # Fit Cox model with top features
  formula_str <- paste("surv_obj ~", paste(top_features, collapse = " + "))
  cox_model <- coxph(as.formula(formula_str), data = data_clean)
  
  # Calculate risk score (linear predictor)
  data_clean$risk_score <- predict(cox_model, type = "lp")
  
  # Categorize risk scores into quartiles
  risk_quartiles <- quantile(data_clean$risk_score, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE)
  data_clean$risk_group <- cut(data_clean$risk_score, 
                               breaks = risk_quartiles,
                               labels = c("Low", "Medium-Low", "Medium-High", "High"),
                               include.lowest = TRUE)
  
  # Survival analysis by risk group
  library(survminer)
  km_by_risk <- survfit(surv_obj ~ risk_group, data = data_clean)
  
  # Log-rank test
  logrank_risk <- survdiff(surv_obj ~ risk_group, data = data_clean)
  pval_risk <- 1 - pchisq(logrank_risk$chisq, length(logrank_risk$n) - 1)
  
  cat("\n--- Risk Group Summary ---\n")
  risk_summary <- data_clean %>%
    group_by(risk_group) %>%
    summarise(
      n = n(),
      mean_risk_score = mean(risk_score),
      event_rate = mean(event_occurred),
      .groups = "drop"
    )
  print(risk_summary)
  
  cat("\nLog-rank test p-value:", format_pvalue(pval_risk), "\n")
  
  # Plot survival curves by risk group
  p_risk_km <- ggsurvplot(
    km_by_risk,
    data = data_clean,
    title = paste("Survival by Risk Group:", dataset_name),
    xlab = "Time (hours)",
    ylab = "Survival Probability",
    risk.table = TRUE,
    conf.int = TRUE,
    pval = TRUE,
    pval.method = TRUE,
    surv.median.line = "hv",
    ggtheme = theme_minimal()
  )
  
  save_figure(p_risk_km$plot, 
              paste0("km_curves/km_by_risk_group_", dataset_name, ".png"),
              width = 10, height = 8)
  
  # Distribution of risk scores
  p_risk_dist <- ggplot(data_clean, aes(x = risk_score, fill = factor(event_occurred))) +
    geom_histogram(alpha = 0.7, bins = 50, position = "identity") +
    facet_wrap(~event_occurred, ncol = 1,
               labeller = labeller(event_occurred = c("0" = "Censored", "1" = "Event"))) +
    labs(title = paste("Risk Score Distribution:", dataset_name),
         x = "Risk Score",
         y = "Frequency",
         fill = "Status") +
    theme_minimal()
  
  save_figure(p_risk_dist, 
              paste0("cox_models/risk_score_distribution_", dataset_name, ".png"),
              width = 10, height = 8)
  
  # Save risk scores
  risk_scores_df <- data_clean %>%
    select(patient_id, risk_score, risk_group, event_occurred, time_to_event)
  
  save_table(risk_scores_df, 
             paste0("risk_scores_", dataset_name, ".csv"),
             "model_summaries")
  
  save_table(risk_summary, 
             paste0("risk_group_summary_", dataset_name, ".csv"),
             "model_summaries")
  
  return(list(
    risk_scores = risk_scores_df,
    risk_summary = risk_summary,
    km_by_risk = km_by_risk,
    logrank_pvalue = pval_risk
  ))
}

# Perform analysis for both datasets
cat("\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Feature Importance Analysis\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

importance_3year <- calculate_feature_importance(data_3year, "3year", model_type = "multivariate")
importance_10year <- calculate_feature_importance(data_10year, "10year", model_type = "multivariate")

risk_3year <- develop_risk_score(data_3year, "3year", top_n = 10)
risk_10year <- develop_risk_score(data_10year, "10year", top_n = 10)

cat("\nFeature importance analysis complete!\n")
cat("Results saved to results/reports/model_summaries/\n")
cat("Figures saved to results/figures/cox_models/ and results/figures/km_curves/\n")
