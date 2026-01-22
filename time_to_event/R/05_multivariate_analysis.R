# Advanced Multivariate Cox Analysis
# This script performs feature selection, model comparison, and interaction analysis

# Load required libraries
library(survival)
library(glmnet)
library(ggplot2)
library(dplyr)
# Load MASS after dplyr to avoid select conflict, use MASS::select explicitly
library(MASS)

# Source utility functions
source("R/utils.R")

# Set project directory
project_root <- get_project_root()
setwd(project_root)

# Load data (assumes 01_load_data.R has been run)
if (!exists("data_3year") || !exists("data_10year")) {
  source("R/01_load_data.R")
}

# Function to perform stepwise feature selection
perform_stepwise_selection <- function(data_obj, dataset_name, direction = "both") {
  cat("\n=== Stepwise Feature Selection:", dataset_name, "===\n")
  
  data <- data_obj$data
  feature_names <- data_obj$feature_names
  
  # Get univariate results to select initial features
  # Check if function exists, if not source it
  if (!exists("perform_univariate_cox")) {
    source("R/04_cox_ph.R", local = TRUE)
  }
  univariate_results <- perform_univariate_cox(data_obj, dataset_name)
  
  # Select features with p < 0.1 for initial model
  initial_features <- univariate_results$results[
    univariate_results$results$pvalue < 0.1, "feature"
  ]
  # Drop NAs and keep only features present in data
  initial_features <- initial_features[!is.na(initial_features)]
  initial_features <- intersect(initial_features, feature_names)
  
  if (length(initial_features) > 50) {
    initial_features <- head(initial_features, 50)  # Limit to top 50
  }
  
  cat("Initial features (p < 0.1):", length(initial_features), "\n")
  if (length(initial_features) == 0) {
    cat("No eligible features for stepwise selection; skipping.\n")
    return(NULL)
  }
  
  # Prepare data
  data_clean <- data %>%
    dplyr::select(all_of(c("surv_obj", initial_features))) %>%
    na.omit()
  
  cat("Observations:", nrow(data_clean), "\n")
  
  # Full model
  formula_full <- as.formula(paste("surv_obj ~", paste(initial_features, collapse = " + ")))
  cox_full <- coxph(formula_full, data = data_clean)
  
  # Null model
  cox_null <- coxph(surv_obj ~ 1, data = data_clean)
  
  # Stepwise selection
  cat("\n--- Performing Stepwise Selection ---\n")
  cox_stepwise <- stepAIC(cox_null, 
                         scope = list(lower = cox_null, upper = cox_full),
                         direction = direction,
                         trace = FALSE)
  
  # Summary
  stepwise_summary <- summary(cox_stepwise)
  print(stepwise_summary)
  
  # Extract selected features
  selected_features <- names(cox_stepwise$coefficients)
  cat("\nSelected features:", length(selected_features), "\n")
  cat(paste(selected_features, collapse = ", "), "\n")
  
  # Model comparison
  cat("\n--- Model Comparison ---\n")
  comparison <- data.frame(
    Model = c("Null", "Full", "Stepwise"),
    AIC = c(AIC(cox_null), AIC(cox_full), AIC(cox_stepwise)),
    BIC = c(BIC(cox_null), BIC(cox_full), BIC(cox_stepwise)),
    n_features = c(0, length(initial_features), length(selected_features))
  )
  print(comparison)
  
  # Save results
  save_table(comparison, 
             paste0("model_comparison_stepwise_", dataset_name, ".csv"),
             "model_summaries")
  
  return(list(
    model = cox_stepwise,
    selected_features = selected_features,
    comparison = comparison
  ))
}

# Function to perform LASSO feature selection
perform_lasso_selection <- function(data_obj, dataset_name) {
  cat("\n=== LASSO Feature Selection:", dataset_name, "===\n")
  
  data <- data_obj$data
  feature_names <- data_obj$feature_names
  
  # Prepare data
  data_clean <- data %>%
    dplyr::select(all_of(c("surv_obj", feature_names))) %>%
    na.omit()
  
  cat("Observations:", nrow(data_clean), "\n")
  cat("Features:", length(feature_names), "\n")
  
  # Prepare X and y for glmnet
  X <- as.matrix(data_clean[, feature_names])
  y <- data_clean$surv_obj
  
  # Fit LASSO with cross-validation
  cat("\n--- Fitting LASSO with Cross-Validation ---\n")
  cv_lasso <- cv.glmnet(X, y, family = "cox", alpha = 1, nfolds = 5)
  
  # Get lambda values
  lambda_min <- cv_lasso$lambda.min
  lambda_1se <- cv_lasso$lambda.1se
  
  cat("Lambda min:", lambda_min, "\n")
  cat("Lambda 1se:", lambda_1se, "\n")
  
  # Get coefficients at lambda.min
  coef_min <- coef(cv_lasso, s = "lambda.min")
  selected_min <- rownames(coef_min)[coef_min[, 1] != 0]
  cat("\nFeatures selected at lambda.min:", length(selected_min), "\n")
  
  # Get coefficients at lambda.1se
  coef_1se <- coef(cv_lasso, s = "lambda.1se")
  selected_1se <- rownames(coef_1se)[coef_1se[, 1] != 0]
  cat("Features selected at lambda.1se:", length(selected_1se), "\n")
  
  # Plot CV curve
  p_cv <- plot(cv_lasso)
  title(paste("LASSO Cross-Validation:", dataset_name))
  
  # Save plot
  png(file.path(get_results_dir(), "figures", "cox_models", 
                paste0("lasso_cv_", dataset_name, ".png")),
      width = 10, height = 7, units = "in", res = 300)
  plot(cv_lasso)
  title(paste("LASSO Cross-Validation:", dataset_name))
  dev.off()
  
  # Fit final Cox model with selected features (lambda.1se for sparsity)
  if (length(selected_1se) > 0) {
    data_final <- data_clean[, c("surv_obj", selected_1se)]
    formula_lasso <- as.formula(paste("surv_obj ~", paste(selected_1se, collapse = " + ")))
    cox_lasso <- coxph(formula_lasso, data = data_final)
    
    lasso_summary <- summary(cox_lasso)
    print(lasso_summary)
    
    # Extract results
    lasso_coef <- lasso_summary$coefficients
    lasso_results <- data.frame(
      feature = rownames(lasso_coef),
      coef = lasso_coef[, "coef"],
      hr = exp(lasso_coef[, "coef"]),
      hr_lower = exp(lasso_coef[, "coef"] - 1.96 * lasso_coef[, "se(coef)"]),
      hr_upper = exp(lasso_coef[, "coef"] + 1.96 * lasso_coef[, "se(coef)"]),
      se = lasso_coef[, "se(coef)"],
      z = lasso_coef[, "z"],
      pvalue = lasso_coef[, "Pr(>|z|)"]
    )
    
    save_table(lasso_results, 
               paste0("cox_lasso_", dataset_name, ".csv"),
               "cox_results")
    
    return(list(
      cv_model = cv_lasso,
      selected_features_min = selected_min,
      selected_features_1se = selected_1se,
      final_model = cox_lasso,
      results = lasso_results,
      lambda_min = lambda_min,
      lambda_1se = lambda_1se
    ))
  } else {
    cat("No features selected by LASSO\n")
    return(NULL)
  }
}

# Function to test interaction terms
test_interactions <- function(data_obj, dataset_name, top_features = 10) {
  cat("\n=== Interaction Analysis:", dataset_name, "===\n")
  
  # Get top features from univariate analysis
  # Check if function exists, if not source it
  if (!exists("perform_univariate_cox")) {
    source("R/04_cox_ph.R", local = TRUE)
  }
  univariate_results <- perform_univariate_cox(data_obj, dataset_name)
  top_features_list <- head(univariate_results$results, top_features)$feature
  
  cat("Testing interactions among top", top_features, "features\n")
  
  data <- data_obj$data
  
  # Prepare data
  data_clean <- data %>%
    dplyr::select(all_of(c("surv_obj", top_features_list))) %>%
    na.omit()
  
  # Test a few key interactions (to avoid combinatorial explosion)
  # Test interactions between top 3 features
  if (length(top_features_list) >= 3) {
    interactions_to_test <- c(
      paste(top_features_list[1], top_features_list[2], sep = ":"),
      paste(top_features_list[1], top_features_list[3], sep = ":"),
      paste(top_features_list[2], top_features_list[3], sep = ":")
    )
    
    interaction_results <- data.frame(
      interaction = character(),
      coef = numeric(),
      hr = numeric(),
      pvalue = numeric(),
      stringsAsFactors = FALSE
    )
    
    for (interaction in interactions_to_test) {
      formula_str <- paste("surv_obj ~", paste(top_features_list, collapse = " + "), 
                          "+", interaction)
      cox_interaction <- tryCatch({
        coxph(as.formula(formula_str), data = data_clean)
      }, error = function(e) {
        return(NULL)
      })
      
      if (!is.null(cox_interaction)) {
        cox_summary <- summary(cox_interaction)
        interaction_coef <- cox_summary$coefficients[interaction, ]
        
        interaction_results <- rbind(interaction_results, data.frame(
          interaction = interaction,
          coef = interaction_coef["coef"],
          hr = exp(interaction_coef["coef"]),
          pvalue = interaction_coef["Pr(>|z|)"]
        ))
      }
    }
    
    if (nrow(interaction_results) > 0) {
      interaction_results$significant <- ifelse(interaction_results$pvalue < 0.05, 
                                                "Yes", "No")
      print(interaction_results)
      
      save_table(interaction_results, 
                 paste0("interactions_", dataset_name, ".csv"),
                 "cox_results")
      
      return(interaction_results)
    }
  }
  
  return(NULL)
}

# Perform analysis for both datasets
cat("\n")
cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Advanced Multivariate Analysis\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# Stepwise selection
stepwise_3year <- perform_stepwise_selection(data_3year, "3year")
stepwise_10year <- perform_stepwise_selection(data_10year, "10year")

# LASSO selection
lasso_3year <- perform_lasso_selection(data_3year, "3year")
lasso_10year <- perform_lasso_selection(data_10year, "10year")

# Interaction analysis
interactions_3year <- test_interactions(data_3year, "3year", top_features = 10)
interactions_10year <- test_interactions(data_10year, "10year", top_features = 10)

cat("\nAdvanced multivariate analysis complete!\n")
cat("Results saved to results/reports/cox_results/ and results/reports/model_summaries/\n")
