# Run Stepwise and LASSO analyses and add to model comparison
# This script runs Stepwise and LASSO Cox models and adds them to the comparison table

# Load required libraries
library(survival)
library(dplyr)
library(MASS)
library(glmnet)

# Source utility functions
source("R/utils.R")

# Set project directory
project_root <- get_project_root()
setwd(project_root)

# Load data if not already loaded
if (!exists("data_3year") || !exists("data_10year")) {
  source("R/01_load_data.R")
}

# Source the multivariate analysis script
source("R/05_multivariate_analysis.R", local = TRUE)

# Function to extract model metrics
extract_model_metrics <- function(cox_model, model_name, dataset_name, n_features) {
  if (is.null(cox_model)) {
    return(NULL)
  }
  
  summary_model <- summary(cox_model)
  
  return(data.frame(
    model = model_name,
    dataset = dataset_name,
    concordance = summary_model$concordance[1],
    aic = AIC(cox_model),
    bic = BIC(cox_model),
    n_features = n_features,
    n_observations = nrow(cox_model$y),
    stringsAsFactors = FALSE
  ))
}

# Function to load existing comparison and add new models
update_model_comparison <- function() {
  cat("\n=== Running Stepwise and LASSO Analyses ===\n")
  
  # Load existing comparison
  comparison_3year_path <- file.path(get_results_dir(), "reports", "model_summaries", "model_comparison_all_3year.csv")
  comparison_10year_path <- file.path(get_results_dir(), "reports", "model_summaries", "model_comparison_all_10year.csv")
  
  # Initialize with existing data if available
  all_metrics <- data.frame()
  
  if (file.exists(comparison_3year_path) && file.exists(comparison_10year_path)) {
    existing_3year <- read.csv(comparison_3year_path, stringsAsFactors = FALSE)
    existing_10year <- read.csv(comparison_10year_path, stringsAsFactors = FALSE)
    all_metrics <- rbind(existing_3year, existing_10year)
    cat("Loaded existing comparison table\n")
  }
  
  # 1. Run Stepwise for 3-year dataset
  cat("\n--- Running Stepwise for 3-year dataset ---\n")
  tryCatch({
    stepwise_3year <- perform_stepwise_selection(data_3year, "3year")
    if (!is.null(stepwise_3year$model)) {
      stepwise_metrics_3year <- extract_model_metrics(
        stepwise_3year$model, 
        "Stepwise Selection", 
        "3-year",
        length(stepwise_3year$selected_features)
      )
      if (!is.null(stepwise_metrics_3year)) {
        all_metrics <- rbind(all_metrics, stepwise_metrics_3year)
        cat("✓ Stepwise 3-year completed\n")
      }
    }
  }, error = function(e) {
    cat("Error in Stepwise 3-year:", e$message, "\n")
  })
  
  # 2. Run Stepwise for 10-year dataset
  cat("\n--- Running Stepwise for 10-year dataset ---\n")
  tryCatch({
    stepwise_10year <- perform_stepwise_selection(data_10year, "10year")
    if (!is.null(stepwise_10year$model)) {
      stepwise_metrics_10year <- extract_model_metrics(
        stepwise_10year$model, 
        "Stepwise Selection", 
        "10-year",
        length(stepwise_10year$selected_features)
      )
      if (!is.null(stepwise_metrics_10year)) {
        all_metrics <- rbind(all_metrics, stepwise_metrics_10year)
        cat("✓ Stepwise 10-year completed\n")
      }
    }
  }, error = function(e) {
    cat("Error in Stepwise 10-year:", e$message, "\n")
  })
  
  # 3. Run LASSO for 3-year dataset
  cat("\n--- Running LASSO for 3-year dataset ---\n")
  tryCatch({
    lasso_3year <- perform_lasso_selection(data_3year, "3year")
    if (!is.null(lasso_3year$final_model)) {
      lasso_metrics_3year <- extract_model_metrics(
        lasso_3year$final_model, 
        "LASSO Regularization", 
        "3-year",
        length(lasso_3year$selected_features_1se)
      )
      if (!is.null(lasso_metrics_3year)) {
        all_metrics <- rbind(all_metrics, lasso_metrics_3year)
        cat("✓ LASSO 3-year completed\n")
      }
    }
  }, error = function(e) {
    cat("Error in LASSO 3-year:", e$message, "\n")
  })
  
  # 4. Run LASSO for 10-year dataset
  cat("\n--- Running LASSO for 10-year dataset ---\n")
  tryCatch({
    lasso_10year <- perform_lasso_selection(data_10year, "10year")
    if (!is.null(lasso_10year$final_model)) {
      lasso_metrics_10year <- extract_model_metrics(
        lasso_10year$final_model, 
        "LASSO Regularization", 
        "10-year",
        length(lasso_10year$selected_features_1se)
      )
      if (!is.null(lasso_metrics_10year)) {
        all_metrics <- rbind(all_metrics, lasso_metrics_10year)
        cat("✓ LASSO 10-year completed\n")
      }
    }
  }, error = function(e) {
    cat("Error in LASSO 10-year:", e$message, "\n")
  })
  
  # Format results
  all_metrics$concordance <- round(all_metrics$concordance, 4)
  all_metrics$aic <- round(all_metrics$aic, 2)
  all_metrics$bic <- round(all_metrics$bic, 2)
  
  # Create updated comparison tables
  comparison_3year <- all_metrics %>%
    filter(dataset == "3-year") %>%
    arrange(desc(concordance))
  
  comparison_10year <- all_metrics %>%
    filter(dataset == "10-year") %>%
    arrange(desc(concordance))
  
  # Side-by-side comparison
  comparison_side_by_side <- all_metrics %>%
    select(model, dataset, concordance, aic, bic, n_features) %>%
    tidyr::pivot_wider(
      names_from = dataset,
      values_from = c(concordance, aic, bic, n_features),
      names_sep = "_"
    )
  
  # Save updated tables
  save_table(comparison_3year, "model_comparison_all_3year.csv", "model_summaries")
  save_table(comparison_10year, "model_comparison_all_10year.csv", "model_summaries")
  save_table(comparison_side_by_side, "model_comparison_all_side_by_side.csv", "model_summaries")
  
  cat("\n=== Updated Model Comparison Tables ===\n")
  cat("\n3-year Dataset:\n")
  print(comparison_3year)
  cat("\n10-year Dataset:\n")
  print(comparison_10year)
  cat("\nSide-by-Side:\n")
  print(comparison_side_by_side)
  
  return(list(
    all_metrics = all_metrics,
    comparison_3year = comparison_3year,
    comparison_10year = comparison_10year,
    comparison_side_by_side = comparison_side_by_side
  ))
}

# Run the update
results <- update_model_comparison()

cat("\n=== Stepwise and LASSO Models Added to Comparison ===\n")
cat("Tables saved to: results/reports/model_summaries/\n")
