# Kaplan-Meier Survival Analysis
# This script performs Kaplan-Meier survival curve analysis

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

# Function to perform Kaplan-Meier analysis
perform_km_analysis <- function(data_obj, dataset_name) {
  cat("\n=== Kaplan-Meier Analysis:", dataset_name, "===\n")
  
  data <- data_obj$data
  
  # 1. Overall survival curve
  cat("\n--- Overall Survival Curve ---\n")
  km_fit_overall <- survfit(surv_obj ~ 1, data = data)
  print(summary(km_fit_overall))
  
  # Plot overall survival
  p_overall <- ggsurvplot(
    km_fit_overall,
    data = data,
    title = paste("Overall Survival Curve:", dataset_name),
    xlab = "Time (hours)",
    ylab = "Survival Probability",
    risk.table = TRUE,
    conf.int = TRUE,
    pval = FALSE,
    surv.median.line = "hv",
    ggtheme = theme_minimal()
  )
  
  save_figure(p_overall$plot, 
              paste0("km_curves/km_overall_", dataset_name, ".png"),
              width = 10, height = 8)
  
  # 2. Survival by event status (for validation)
  # This should show clear separation
  data$event_factor <- factor(data$event_occurred, 
                              levels = c(0, 1), 
                              labels = c("Censored", "Event"))
  
  km_fit_by_event <- survfit(surv_obj ~ event_factor, data = data)
  
  # Log-rank test
  logrank_test <- survdiff(surv_obj ~ event_factor, data = data)
  pval_logrank <- 1 - pchisq(logrank_test$chisq, length(logrank_test$n) - 1)
  
  cat("\n--- Log-Rank Test (Event vs Censored) ---\n")
  cat("Chi-square:", logrank_test$chisq, "\n")
  cat("P-value:", format_pvalue(pval_logrank), "\n")
  
  p_by_event <- ggsurvplot(
    km_fit_by_event,
    data = data,
    title = paste("Survival by Event Status:", dataset_name),
    xlab = "Time (hours)",
    ylab = "Survival Probability",
    risk.table = TRUE,
    conf.int = TRUE,
    pval = TRUE,
    pval.method = TRUE,
    surv.median.line = "hv",
    legend.title = "Status",
    legend.labs = c("Censored", "Event"),
    ggtheme = theme_minimal()
  )
  
  save_figure(p_by_event$plot, 
              paste0("km_curves/km_by_event_", dataset_name, ".png"),
              width = 10, height = 8)
  
  # 3. Stratified analysis by key features
  # Get numeric features for stratification
  numeric_features <- data %>%
    select(-patient_id, -measurement_time, -event_time, -time_to_event, 
           -event_occurred, -surv_obj, -event_factor) %>%
    select_if(is.numeric) %>%
    colnames()
  
  # Select top features for stratification (based on variance or importance)
  # For now, use first few features as examples
  features_to_stratify <- head(numeric_features, 5)
  
  km_results <- list()
  km_results$overall <- km_fit_overall
  km_results$by_event <- km_fit_by_event
  
  # Stratified analysis for each feature
  for (feature in features_to_stratify) {
    cat("\n--- Stratified by", feature, "---\n")
    
    # Initialize variables for this iteration
    strat_var <- NULL
    formula_str <- NULL
    
    # Create tertiles for continuous features
    feature_values <- data[[feature]]
    unique_vals <- unique(feature_values[!is.na(feature_values)])
    group_col_name <- paste0("group_", gsub("[^A-Za-z0-9]", "_", feature))
    
    if (length(unique_vals) > 3) {
      tertiles <- quantile(feature_values, probs = c(0, 1/3, 2/3, 1), na.rm = TRUE)
      # Check if breaks are unique
      if (length(unique(tertiles)) < 4) {
        cat("Skipping", feature, "- insufficient variation for tertiles\n")
        next
      }
      data[[group_col_name]] <- tryCatch({
        cut(feature_values, breaks = tertiles, labels = c("Low", "Medium", "High"), include.lowest = TRUE)
      }, error = function(e) {
        cat("Error creating groups for", feature, ":", e$message, "\n")
        return(NULL)
      })
      if (is.null(data[[group_col_name]])) next
      strat_var <- group_col_name
    } else if (length(unique_vals) >= 2) {
      # Use as categorical if few unique values
      data[[group_col_name]] <- factor(feature_values)
      strat_var <- group_col_name
    } else {
      cat("Skipping", feature, "- insufficient unique values (only", length(unique_vals), ")\n")
      next
    }
    
    # Check if we have multiple groups (only if strat_var was set)
    if (!exists("strat_var") || is.null(strat_var)) {
      cat("Skipping", feature, "- could not create stratification variable\n")
      next
    }
    
    n_groups <- length(unique(data[[strat_var]][!is.na(data[[strat_var]])]))
    if (n_groups < 2) {
      cat("Skipping", feature, "- insufficient groups (only", n_groups, "group(s))\n")
      next
    }
    
    # Fit KM model
    formula_str <- paste("surv_obj ~", strat_var)
    km_fit_strat <- tryCatch({
      survfit(as.formula(formula_str), data = data)
    }, error = function(e) {
      cat("Error fitting KM model for", feature, ":", e$message, "\n")
      return(NULL)
    })
    
    if (is.null(km_fit_strat)) next
    
    # Log-rank test
    logrank_strat <- tryCatch({
      survdiff(as.formula(formula_str), data = data)
    }, error = function(e) {
      cat("Error in log-rank test for", feature, ":", e$message, "\n")
      return(NULL)
    })
    
    if (is.null(logrank_strat)) next
    
    pval_strat <- 1 - pchisq(logrank_strat$chisq, length(logrank_strat$n) - 1)
    
    cat("Log-rank test p-value:", format_pvalue(pval_strat), "\n")
    
    # Plot
    tryCatch({
      p_strat <- ggsurvplot(
        km_fit_strat,
        data = data,
        title = paste("Survival by", feature, ":", dataset_name),
        xlab = "Time (hours)",
        ylab = "Survival Probability",
        risk.table = TRUE,
        conf.int = TRUE,
        pval = TRUE,
        pval.method = TRUE,
        surv.median.line = "hv",
        ggtheme = theme_minimal()
      )
      
      # Clean feature name for filename
      feature_clean <- gsub("[^A-Za-z0-9]", "_", feature)
      save_figure(p_strat$plot, 
                  paste0("km_curves/km_", feature_clean, "_", dataset_name, ".png"),
                  width = 10, height = 8)
    }, error = function(e) {
      cat("Error creating plot for", feature, ":", e$message, "\n")
      # Try a simpler plot
      tryCatch({
        p_simple <- ggsurvplot(km_fit_strat, data = data, ggtheme = theme_minimal())
        feature_clean <- gsub("[^A-Za-z0-9]", "_", feature)
        save_figure(p_simple$plot, 
                    paste0("km_curves/km_", feature_clean, "_", dataset_name, ".png"),
                    width = 10, height = 8)
      }, error = function(e2) {
        cat("Could not create plot for", feature, "\n")
      })
    })
    
    km_results[[feature]] <- list(
      fit = km_fit_strat,
      logrank = logrank_strat,
      pvalue = pval_strat
    )
  }
  
  # 4. Median survival times
  cat("\n--- Median Survival Times ---\n")
  median_survival <- summary(km_fit_overall)$table
  print(median_survival)
  
  # 5. Survival probabilities at key time points
  cat("\n--- Survival Probabilities at Key Time Points ---\n")
  time_points <- c(24, 48, 72, 168, 720)  # 1 day, 2 days, 3 days, 1 week, 1 month (hours)
  
  surv_probs <- summary(km_fit_overall, times = time_points, extend = TRUE)
  surv_table <- data.frame(
    time_hours = surv_probs$time,
    time_days = surv_probs$time / 24,
    survival_prob = surv_probs$surv,
    lower_ci = surv_probs$lower,
    upper_ci = surv_probs$upper
  )
  
  print(surv_table)
  
  # Save results
  save_table(surv_table, 
             paste0("survival_probabilities_", dataset_name, ".csv"),
             "descriptive_stats")
  
  return(list(
    km_fit_overall = km_fit_overall,
    km_fit_by_event = km_fit_by_event,
    stratified_results = km_results,
    median_survival = median_survival,
    survival_probabilities = surv_table
  ))
}

# Perform analysis for both datasets
cat("\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Kaplan-Meier Survival Analysis\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

km_3year <- perform_km_analysis(data_3year, "3year")
km_10year <- perform_km_analysis(data_10year, "10year")

cat("\nKaplan-Meier analysis complete!\n")
cat("Results saved to results/reports/descriptive_stats/\n")
cat("Figures saved to results/figures/km_curves/\n")
