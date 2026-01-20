# Model Comparison Analysis
# This script creates comprehensive comparison tables and visualizations for all models

# Load required libraries
library(survival)
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Source utility functions
source("R/utils.R")

# Set project directory
project_root <- get_project_root()
setwd(project_root)

# Load data if not already loaded
if (!exists("data_3year") || !exists("data_10year")) {
  source("R/01_load_data.R")
}

# Function to extract model performance metrics
extract_model_metrics <- function(cox_model, model_name, dataset_name, n_features, n_obs = NA) {
  if (is.null(cox_model)) {
    return(data.frame(
      model = model_name,
      dataset = dataset_name,
      concordance = NA,
      aic = NA,
      bic = NA,
      n_features = n_features,
      n_observations = n_obs,
      stringsAsFactors = FALSE
    ))
  }
  
  summary_model <- summary(cox_model)
  
  return(data.frame(
    model = model_name,
    dataset = dataset_name,
    concordance = summary_model$concordance[1],
    aic = AIC(cox_model),
    bic = BIC(cox_model),
    n_features = n_features,
    n_observations = ifelse(is.na(n_obs), nrow(cox_model$y), n_obs),
    stringsAsFactors = FALSE
  ))
}

# Function to create comprehensive model comparison
create_model_comparison <- function() {
  cat("\n=== Creating Comprehensive Model Comparison Tables ===\n")
  
  # Initialize results
  all_metrics <- data.frame()
  
  # 1. Univariate Models (using top feature as representative)
  cat("\n--- Extracting Univariate Model Metrics ---\n")
  
  # Load univariate results
  univariate_3year <- read.csv(
    file.path(get_results_dir(), "reports", "cox_results", "cox_univariate_3year.csv"),
    stringsAsFactors = FALSE
  )
  univariate_10year <- read.csv(
    file.path(get_results_dir(), "reports", "cox_results", "cox_univariate_10year.csv"),
    stringsAsFactors = FALSE
  )
  
  # Get top feature and fit model for metrics
  top_feature_3year <- univariate_3year$feature[1]
  top_feature_10year <- univariate_10year$feature[1]
  
  # Fit univariate models for metrics
  data_3year_clean <- data_3year$data %>%
    select(all_of(c("surv_obj", top_feature_3year))) %>%
    na.omit()
  
  data_10year_clean <- data_10year$data %>%
    select(all_of(c("surv_obj", top_feature_10year))) %>%
    na.omit()
  
  cox_uni_3year <- coxph(as.formula(paste("surv_obj ~", top_feature_3year)), 
                         data = data_3year_clean)
  cox_uni_10year <- coxph(as.formula(paste("surv_obj ~", top_feature_10year)), 
                          data = data_10year_clean)
  
  all_metrics <- rbind(
    all_metrics,
    extract_model_metrics(cox_uni_3year, "Univariate (Top Feature)", "3-year", 1),
    extract_model_metrics(cox_uni_10year, "Univariate (Top Feature)", "10-year", 1)
  )
  
  # 2. Multivariate Models
  cat("\n--- Extracting Multivariate Model Metrics ---\n")
  
  # Load multivariate results to get features
  multivariate_3year <- read.csv(
    file.path(get_results_dir(), "reports", "cox_results", "cox_multivariate_3year.csv"),
    stringsAsFactors = FALSE
  )
  multivariate_10year <- read.csv(
    file.path(get_results_dir(), "reports", "cox_results", "cox_multivariate_10year.csv"),
    stringsAsFactors = FALSE
  )
  
  features_3year <- multivariate_3year$feature
  features_10year <- multivariate_10year$feature
  
  # Fit multivariate models
  data_3year_multi <- data_3year$data %>%
    select(all_of(c("surv_obj", features_3year))) %>%
    na.omit()
  
  data_10year_multi <- data_10year$data %>%
    select(all_of(c("surv_obj", features_10year))) %>%
    na.omit()
  
  formula_3year <- as.formula(paste("surv_obj ~", paste(features_3year, collapse = " + ")))
  formula_10year <- as.formula(paste("surv_obj ~", paste(features_10year, collapse = " + ")))
  
  cox_multi_3year <- coxph(formula_3year, data = data_3year_multi)
  cox_multi_10year <- coxph(formula_10year, data = data_10year_multi)
  
  all_metrics <- rbind(
    all_metrics,
    extract_model_metrics(cox_multi_3year, "Multivariate (Top 20)", "3-year", length(features_3year)),
    extract_model_metrics(cox_multi_10year, "Multivariate (Top 20)", "10-year", length(features_10year))
  )
  
  # 3. Stepwise Models
  cat("\n--- Extracting Stepwise Model Metrics ---\n")
  
  stepwise_files <- list.files(
    file.path(get_results_dir(), "reports", "model_summaries"),
    pattern = "model_comparison_stepwise",
    full.names = TRUE
  )
  
  if (length(stepwise_files) > 0) {
    for (file in stepwise_files) {
      if (grepl("3year", file)) {
        stepwise_comp <- read.csv(file, stringsAsFactors = FALSE)
        if ("Stepwise" %in% stepwise_comp$Model) {
          stepwise_row <- stepwise_comp[stepwise_comp$Model == "Stepwise", ]
          # Try to get actual model to calculate concordance
          # For now, use AIC/BIC from comparison
          all_metrics <- rbind(
            all_metrics,
            data.frame(
              model = "Stepwise Selection",
              dataset = "3-year",
              concordance = NA,  # Will calculate if model available
              aic = stepwise_row$AIC,
              bic = stepwise_row$BIC,
              n_features = stepwise_row$n_features,
              n_observations = NA,
              stringsAsFactors = FALSE
            )
          )
        }
      } else if (grepl("10year", file)) {
        stepwise_comp <- read.csv(file, stringsAsFactors = FALSE)
        if ("Stepwise" %in% stepwise_comp$Model) {
          stepwise_row <- stepwise_comp[stepwise_comp$Model == "Stepwise", ]
          all_metrics <- rbind(
            all_metrics,
            data.frame(
              model = "Stepwise Selection",
              dataset = "10-year",
              concordance = NA,
              aic = stepwise_row$AIC,
              bic = stepwise_row$BIC,
              n_features = stepwise_row$n_features,
              n_observations = NA,
              stringsAsFactors = FALSE
            )
          )
        }
      }
    }
  } else {
    cat("Stepwise comparison files not found. Skipping stepwise models for now.\n")
    cat("(Stepwise models can be added later if needed)\n")
  }
  
  # 4. LASSO Models
  cat("\n--- Extracting LASSO Model Metrics ---\n")
  
  lasso_files <- list.files(
    file.path(get_results_dir(), "reports", "cox_results"),
    pattern = "cox_lasso",
    full.names = TRUE
  )
  
  if (length(lasso_files) > 0) {
    for (file in lasso_files) {
      if (grepl("3year", file)) {
        lasso_results <- read.csv(file, stringsAsFactors = FALSE)
        # Fit model to get metrics
        features_lasso <- lasso_results$feature
        if (length(features_lasso) > 0) {
          data_lasso <- data_3year$data %>%
            select(all_of(c("surv_obj", features_lasso))) %>%
            na.omit()
          formula_lasso <- as.formula(paste("surv_obj ~", paste(features_lasso, collapse = " + ")))
          cox_lasso <- coxph(formula_lasso, data = data_lasso)
          
          all_metrics <- rbind(
            all_metrics,
            extract_model_metrics(cox_lasso, "LASSO Regularization", "3-year", length(features_lasso))
          )
        }
      } else if (grepl("10year", file)) {
        lasso_results <- read.csv(file, stringsAsFactors = FALSE)
        features_lasso <- lasso_results$feature
        if (length(features_lasso) > 0) {
          data_lasso <- data_10year$data %>%
            select(all_of(c("surv_obj", features_lasso))) %>%
            na.omit()
          formula_lasso <- as.formula(paste("surv_obj ~", paste(features_lasso, collapse = " + ")))
          cox_lasso <- coxph(formula_lasso, data = data_lasso)
          
          all_metrics <- rbind(
            all_metrics,
            extract_model_metrics(cox_lasso, "LASSO Regularization", "10-year", length(features_lasso))
          )
        }
      }
    }
  } else {
    cat("LASSO results not found. Skipping LASSO models for now.\n")
    cat("(LASSO models can be added later if needed)\n")
  }
  
  # Format results
  all_metrics$concordance <- round(all_metrics$concordance, 4)
  all_metrics$aic <- round(all_metrics$aic, 2)
  all_metrics$bic <- round(all_metrics$bic, 2)
  
  # Create comparison tables
  
  # Table 1: All models for 3-year dataset
  comparison_3year <- all_metrics %>%
    filter(dataset == "3-year") %>%
    arrange(desc(concordance))
  
  # Table 2: All models for 10-year dataset
  comparison_10year <- all_metrics %>%
    filter(dataset == "10-year") %>%
    arrange(desc(concordance))
  
  # Table 3: Side-by-side comparison
  comparison_side_by_side <- all_metrics %>%
    select(model, dataset, concordance, aic, bic, n_features) %>%
    pivot_wider(
      names_from = dataset,
      values_from = c(concordance, aic, bic, n_features),
      names_sep = "_"
    )
  
  # Save tables
  save_table(comparison_3year, "model_comparison_all_3year.csv", "model_summaries")
  save_table(comparison_10year, "model_comparison_all_10year.csv", "model_summaries")
  save_table(comparison_side_by_side, "model_comparison_all_side_by_side.csv", "model_summaries")
  
  # Print results
  cat("\n=== Model Comparison: 3-year Dataset ===\n")
  print(comparison_3year)
  
  cat("\n=== Model Comparison: 10-year Dataset ===\n")
  print(comparison_10year)
  
  cat("\n=== Side-by-Side Comparison ===\n")
  print(comparison_side_by_side)
  
  return(list(
    all_metrics = all_metrics,
    comparison_3year = comparison_3year,
    comparison_10year = comparison_10year,
    comparison_side_by_side = comparison_side_by_side
  ))
}

# Function to create visualization
visualize_model_comparison <- function(comparison_results) {
  cat("\n=== Creating Model Comparison Visualizations ===\n")
  
  all_metrics <- comparison_results$all_metrics
  
  # Remove rows with missing concordance for visualization
  metrics_vis <- all_metrics %>%
    filter(!is.na(concordance))
  
  # 1. Concordance comparison
  p1 <- ggplot(metrics_vis, aes(x = model, y = concordance, fill = dataset)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    geom_text(aes(label = round(concordance, 3)), 
              position = position_dodge(width = 0.9), 
              vjust = -0.5, size = 3) +
    labs(title = "Model Comparison: Concordance (C-index)",
         x = "Model Type",
         y = "Concordance (C-index)",
         fill = "Dataset") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5, face = "bold")) +
    ylim(0, max(metrics_vis$concordance, na.rm = TRUE) * 1.1)
  
  save_figure(p1, "comparisons/model_concordance_comparison.png", width = 12, height = 7)
  
  # 2. AIC comparison
  p2 <- ggplot(metrics_vis, aes(x = model, y = aic, fill = dataset)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    geom_text(aes(label = round(aic, 0)), 
              position = position_dodge(width = 0.9), 
              vjust = -0.5, size = 3) +
    labs(title = "Model Comparison: AIC (Lower is Better)",
         x = "Model Type",
         y = "AIC",
         fill = "Dataset") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5, face = "bold"))
  
  save_figure(p2, "comparisons/model_aic_comparison.png", width = 12, height = 7)
  
  # 3. Number of features comparison
  p3 <- ggplot(metrics_vis, aes(x = model, y = n_features, fill = dataset)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    geom_text(aes(label = n_features), 
              position = position_dodge(width = 0.9), 
              vjust = -0.5, size = 3) +
    labs(title = "Model Comparison: Number of Features",
         x = "Model Type",
         y = "Number of Features",
         fill = "Dataset") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5, face = "bold"))
  
  save_figure(p3, "comparisons/model_features_comparison.png", width = 12, height = 7)
  
  # 4. Comprehensive comparison plot
  # Reshape data for comprehensive plot
  metrics_long <- metrics_vis %>%
    select(model, dataset, concordance, aic, n_features) %>%
    pivot_longer(cols = c(concordance, aic, n_features), 
                 names_to = "metric", 
                 values_to = "value")
  
  # Normalize metrics for comparison (0-1 scale)
  metrics_long <- metrics_long %>%
    group_by(metric) %>%
    mutate(value_norm = ifelse(metric == "aic", 
                              1 - (value - min(value, na.rm = TRUE)) / (max(value, na.rm = TRUE) - min(value, na.rm = TRUE)),
                              (value - min(value, na.rm = TRUE)) / (max(value, na.rm = TRUE) - min(value, na.rm = TRUE))))
  
  p4 <- ggplot(metrics_long, aes(x = model, y = value_norm, fill = dataset)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    facet_wrap(~ metric, scales = "free_y", 
               labeller = labeller(metric = c("concordance" = "Concordance (Normalized)",
                                             "aic" = "AIC (Normalized, Lower Better)",
                                             "n_features" = "Number of Features (Normalized)"))) +
    labs(title = "Comprehensive Model Comparison (Normalized Metrics)",
         x = "Model Type",
         y = "Normalized Value",
         fill = "Dataset") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5, face = "bold"),
          strip.text = element_text(face = "bold"))
  
  save_figure(p4, "comparisons/model_comprehensive_comparison.png", width = 14, height = 8)
  
  # 5. Scatter plot: Concordance vs AIC
  p5 <- ggplot(metrics_vis, aes(x = aic, y = concordance, color = dataset, shape = model)) +
    geom_point(size = 4, alpha = 0.7) +
    geom_text(aes(label = model), vjust = -0.5, hjust = 0.5, size = 2.5) +
    labs(title = "Model Performance: Concordance vs AIC",
         x = "AIC (Lower is Better)",
         y = "Concordance (C-index, Higher is Better)",
         color = "Dataset",
         shape = "Model Type") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"),
          legend.position = "right")
  
  save_figure(p5, "comparisons/model_concordance_vs_aic.png", width = 12, height = 8)
  
  # 6. Model ranking visualization
  ranking_data <- metrics_vis %>%
    group_by(dataset) %>%
    arrange(desc(concordance)) %>%
    mutate(rank = row_number()) %>%
    ungroup()
  
  p6 <- ggplot(ranking_data, aes(x = reorder(model, -rank), y = rank, fill = dataset)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    scale_y_reverse(breaks = 1:max(ranking_data$rank)) +
    labs(title = "Model Ranking by Concordance (1 = Best)",
         x = "Model Type",
         y = "Rank (1 = Best Performance)",
         fill = "Dataset") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(hjust = 0.5, face = "bold"))
  
  save_figure(p6, "comparisons/model_ranking.png", width = 12, height = 7)
  
  cat("\n=== Model Comparison Visualizations Created ===\n")
}

# Run comparison
cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("MODEL COMPARISON ANALYSIS\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

model_comparison_results <- create_model_comparison()
visualize_model_comparison(model_comparison_results)

cat("\n=== Model Comparison Tables Created ===\n")
cat("Files saved to: results/reports/model_summaries/\n")
cat("  - model_comparison_all_3year.csv\n")
cat("  - model_comparison_all_10year.csv\n")
cat("  - model_comparison_all_side_by_side.csv\n")
cat("\nFigures saved to: results/figures/comparisons/\n")
cat("  - model_concordance_comparison.png\n")
cat("  - model_aic_comparison.png\n")
cat("  - model_features_comparison.png\n")
cat("  - model_comprehensive_comparison.png\n")
cat("  - model_concordance_vs_aic.png\n")
cat("  - model_ranking.png\n")
