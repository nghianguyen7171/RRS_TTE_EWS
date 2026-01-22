# Update model comparison tables with Stepwise results
# This script extracts Stepwise metrics and adds them to the main comparison tables

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
  
  # We need to re-fit stepwise models to get concordance
  # Load data
  if (!exists("data_3year") || !exists("data_10year")) {
    source("R/01_load_data.R")
  }
  
  # Source multivariate analysis to get stepwise function
  source("R/05_multivariate_analysis.R", local = TRUE)
  
  # Re-run stepwise to get full metrics (this is quick since models are already selected)
  cat("\n=== Extracting Stepwise Metrics ===\n")
  
  # For 3-year: Re-run stepwise
  tryCatch({
    cat("Re-running Stepwise for 3-year to get concordance...\n")
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
      
      # Remove old stepwise if exists
      all_metrics <- all_metrics[!(all_metrics$model == "Stepwise Selection" & all_metrics$dataset == "3-year"), ]
      all_metrics <- rbind(all_metrics, stepwise_3year_metrics)
      cat("✓ Stepwise 3-year metrics extracted\n")
    }
  }, error = function(e) {
    cat("Error extracting Stepwise 3-year:", e$message, "\n")
  })
  
  # For 10-year: Re-run stepwise
  tryCatch({
    cat("Re-running Stepwise for 10-year to get concordance...\n")
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
      
      # Remove old stepwise if exists
      all_metrics <- all_metrics[!(all_metrics$model == "Stepwise Selection" & all_metrics$dataset == "10-year"), ]
      all_metrics <- rbind(all_metrics, stepwise_10year_metrics)
      cat("✓ Stepwise 10-year metrics extracted\n")
    }
  }, error = function(e) {
    cat("Error extracting Stepwise 10-year:", e$message, "\n")
  })
  
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
