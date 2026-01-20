# Master Script: Run All Survival Analyses
# This script runs all R analyses in sequence

cat("\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("R-Based Survival Analysis for Time-to-Event Task\n")
cat("Running Complete Analysis Pipeline\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

# Record start time
start_time <- Sys.time()

# Set project directory - assume script is run from time_to_event directory
# or from project root
current_dir <- getwd()
if (basename(current_dir) != "time_to_event") {
  # Try to navigate to time_to_event
  if (file.exists("time_to_event")) {
    setwd("time_to_event")
  } else if (file.exists(file.path("..", "time_to_event"))) {
    setwd(file.path("..", "time_to_event"))
  }
}

project_root <- getwd()
cat("Project root:", project_root, "\n")
cat("Working directory:", getwd(), "\n\n")

# Source utils first
source("R/utils.R")

# Check required packages
cat("Checking required packages...\n")
if (!check_packages()) {
  stop("Please install missing packages before running analysis")
}
cat("All required packages are installed.\n\n")

# Phase 1: Data Loading
cat("\n", paste(rep("-", 70), collapse = ""), "\n")
cat("PHASE 1: Data Loading\n")
cat(paste(rep("-", 70), collapse = ""), "\n")
source("R/01_load_data.R")

# Phase 2: Descriptive Analysis
cat("\n", paste(rep("-", 70), collapse = ""), "\n")
cat("PHASE 2: Descriptive Analysis\n")
cat(paste(rep("-", 70), collapse = ""), "\n")
source("R/02_descriptive_analysis.R")

# Phase 3: Survival Analysis
cat("\n", paste(rep("-", 70), collapse = ""), "\n")
cat("PHASE 3: Survival Analysis\n")
cat(paste(rep("-", 70), collapse = ""), "\n")

cat("\n3.1: Kaplan-Meier Analysis\n")
source("R/03_kaplan_meier.R")

cat("\n3.2: Cox Proportional Hazards\n")
source("R/04_cox_ph.R")

# Phase 4: Advanced Analysis
cat("\n", paste(rep("-", 70), collapse = ""), "\n")
cat("PHASE 4: Advanced Analysis\n")
cat(paste(rep("-", 70), collapse = ""), "\n")

cat("\n4.1: Multivariate Analysis\n")
source("R/05_multivariate_analysis.R")

cat("\n4.2: Feature Importance\n")
source("R/06_feature_importance.R")

cat("\n4.3: Model Diagnostics\n")
source("R/07_model_diagnostics.R")

# Phase 5: Comparison
cat("\n", paste(rep("-", 70), collapse = ""), "\n")
cat("PHASE 5: Dataset Comparison\n")
cat(paste(rep("-", 70), collapse = ""), "\n")
source("R/08_comparison.R")

# Summary
end_time <- Sys.time()
elapsed_time <- difftime(end_time, start_time, units = "mins")

cat("\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("ANALYSIS COMPLETE!\n")
cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Total elapsed time:", round(elapsed_time, 2), "minutes\n")
cat("\nResults saved to:\n")
cat("  - Figures: results/figures/\n")
cat("  - Reports: results/reports/\n")
cat("  - Summaries: results/summaries/\n")
cat("\n")
