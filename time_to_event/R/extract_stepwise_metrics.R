# Quick extraction of Stepwise metrics from saved comparison files
# This avoids re-running stepwise by using saved AIC/BIC and estimating concordance

library(survival)
library(dplyr)
library(tidyr)

source("R/utils.R")

setwd(get_project_root())

# Load existing comparison
comparison_3year_path <- file.path(get_results_dir(), "reports", "model_summaries", "model_comparison_all_3year.csv")
comparison_10year_path <- file.path(get_results_dir(), "reports", "model_summaries", "model_comparison_all_10year.csv")

# Load existing data
if (file.exists(comparison_3year_path) && file.exists(comparison_10year_path)) {
  existing_3year <- read.csv(comparison_3year_path, stringsAsFactors = FALSE)
  existing_10year <- read.csv(comparison_10year_path, stringsAsFactors = FALSE)
  all_metrics <- rbind(existing_3year, existing_10year)
  cat("Loaded existing comparison table\n")
} else {
  all_metrics <- data.frame()
}

# Load stepwise comparison files
stepwise_3year_path <- file.path(get_results_dir(), "reports", "model_summaries", "model_comparison_stepwise_3year.csv")
stepwise_10year_path <- file.path(get_results_dir(), "reports", "model_summaries", "model_comparison_stepwise_10year.csv")

if (file.exists(stepwise_3year_path) && file.exists(stepwise_10year_path)) {
  stepwise_3year <- read.csv(stepwise_3year_path, stringsAsFactors = FALSE)
  stepwise_10year <- read.csv(stepwise_10year_path, stringsAsFactors = FALSE)
  
  # Get stepwise metrics
  stepwise_3year_row <- stepwise_3year[stepwise_3year$Model == "Stepwise", ]
  stepwise_10year_row <- stepwise_10year[stepwise_10year$Model == "Stepwise", ]
  
  # Load data to get n_observations
  if (!exists("data_3year") || !exists("data_10year")) {
    source("R/01_load_data.R")
  }
  
  # We need concordance - let's try to get it from a quick model fit
  # Or use approximate values from log (10-year had 0.778)
  # For now, let's fit a quick model to get concordance
  
  cat("\n=== Extracting Stepwise Metrics ===\n")
  
  # For 3-year: We'll need to re-fit, but let's try to get selected features first
  # Actually, let's just add with estimated concordance based on AIC relationship
  # Or better: load the saved stepwise model if it exists
  
  # Check log for concordance values
  log_file <- "/tmp/stepwise_lasso_full.log"
  if (file.exists(log_file)) {
    log_content <- readLines(log_file)
    # Look for concordance in log
    concordance_lines <- grep("Concordance=", log_content, value = TRUE)
    cat("Found concordance in log:\n")
    print(concordance_lines)
  }
  
  # For now, let's add stepwise with AIC/BIC from saved files
  # and estimate concordance or mark as "needs calculation"
  
  # Try to get concordance by quick re-fit with known features
  # But that's what we're trying to avoid...
  
  # Alternative: Add stepwise with placeholder concordance, note that it needs full calculation
  # Or: Use the fact that stepwise typically performs between univariate and multivariate
  
  # From the log, 10-year stepwise had Concordance=0.778
  # Let's estimate 3-year based on similar improvement pattern
  
  # Actually, let's just add what we have and note concordance needs to be calculated
  # Or better: do a minimal re-fit just to get concordance
  
  cat("Stepwise 3-year: AIC=", stepwise_3year_row$AIC, ", Features=", stepwise_3year_row$n_features, "\n")
  cat("Stepwise 10-year: AIC=", stepwise_10year_row$AIC, ", Features=", stepwise_10year_row$n_features, "\n")
  
  # From log: 10-year stepwise Concordance = 0.778
  # Estimate 3-year: if multivariate is 0.885 and stepwise has similar AIC improvement,
  # stepwise should be close to multivariate (maybe 0.88-0.89)
  # But let's be conservative and estimate 0.87-0.88
  
  # Actually, let's just re-fit quickly - it's faster than stepwise selection
  source("R/05_multivariate_analysis.R", local = TRUE)
  
  # Quick re-fit for 3-year
  cat("Quick re-fit for 3-year stepwise...\n")
  stepwise_3year_result <- perform_stepwise_selection(data_3year, "3year")
  if (!is.null(stepwise_3year_result$model)) {
    summary_model <- summary(stepwise_3year_result$model)
    stepwise_3year_metrics <- data.frame(
      model = "Stepwise Selection",
      dataset = "3-year",
      concordance = summary_model$concordance[1],
      aic = AIC(stepwise_3year_result$model),
      bic = BIC(stepwise_3year_result$model),
      n_features = length(stepwise_3year_result$selected_features),
      n_observations = nrow(stepwise_3year_result$model$y),
      stringsAsFactors = FALSE
    )
    all_metrics <- all_metrics[!(all_metrics$model == "Stepwise Selection" & all_metrics$dataset == "3-year"), ]
    all_metrics <- rbind(all_metrics, stepwise_3year_metrics)
    cat("✓ Stepwise 3-year: C-index =", stepwise_3year_metrics$concordance, "\n")
  }
  
  # Quick re-fit for 10-year
  cat("Quick re-fit for 10-year stepwise...\n")
  stepwise_10year_result <- perform_stepwise_selection(data_10year, "10year")
  if (!is.null(stepwise_10year_result$model)) {
    summary_model <- summary(stepwise_10year_result$model)
    stepwise_10year_metrics <- data.frame(
      model = "Stepwise Selection",
      dataset = "10-year",
      concordance = summary_model$concordance[1],
      aic = AIC(stepwise_10year_result$model),
      bic = BIC(stepwise_10year_result$model),
      n_features = length(stepwise_10year_result$selected_features),
      n_observations = nrow(stepwise_10year_result$model$y),
      stringsAsFactors = FALSE
    )
    all_metrics <- all_metrics[!(all_metrics$model == "Stepwise Selection" & all_metrics$dataset == "10-year"), ]
    all_metrics <- rbind(all_metrics, stepwise_10year_metrics)
    cat("✓ Stepwise 10-year: C-index =", stepwise_10year_metrics$concordance, "\n")
  }
  
} else {
  cat("Stepwise comparison files not found\n")
}

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

cat("\n✓ Comparison tables updated with Stepwise results\n")
