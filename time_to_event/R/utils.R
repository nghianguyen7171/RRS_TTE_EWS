# Utility functions for survival analysis

# Load numpy arrays using reticulate
load_numpy_array <- function(file_path) {
  if (!require(reticulate, quietly = TRUE)) {
    stop("reticulate package is required. Install with: install.packages('reticulate')")
  }
  
  # Check if numpy is available
  if (!py_module_available("numpy")) {
    py_install("numpy")
  }
  
  np <- import("numpy")
  array_data <- np$load(file_path, allow_pickle = FALSE)
  return(as.vector(array_data))
}

# Alternative: Load numpy arrays using RcppCNPy (if available)
load_numpy_array_cnpy <- function(file_path) {
  if (!require(RcppCNPy, quietly = TRUE)) {
    stop("RcppCNPy package is required. Install with: install.packages('RcppCNPy')")
  }
  
  array_data <- npyLoad(file_path)
  return(as.vector(array_data))
}

# Create survival object from time and event
create_survival_object <- function(time, event) {
  if (!require(survival, quietly = TRUE)) {
    stop("survival package is required")
  }
  
  return(Surv(time = time, event = event))
}

# Get project root directory
get_project_root <- function() {
  # Simple approach: use current working directory
  # Assume scripts are run from time_to_event directory
  wd <- getwd()
  
  # If we're in R/, go up one level
  if (basename(wd) == "R") {
    project_root <- dirname(wd)
  } else if (basename(wd) == "time_to_event") {
    project_root <- wd
  } else {
    # Try to find time_to_event in the path
    parts <- strsplit(wd, "/")[[1]]
    idx <- which(parts == "time_to_event")
    if (length(idx) > 0) {
      project_root <- paste(parts[1:idx], collapse = "/")
    } else {
      # Default: assume we're in time_to_event or should be
      project_root <- wd
    }
  }
  
  return(project_root)
}

# Get data directory path
get_data_dir <- function(dataset = c("3year", "10year")) {
  dataset <- match.arg(dataset)
  project_root <- get_project_root()
  data_dir <- file.path(project_root, "data", paste0("processed_", dataset))
  return(data_dir)
}

# Get results directory path
get_results_dir <- function() {
  project_root <- get_project_root()
  results_dir <- file.path(project_root, "results")
  return(results_dir)
}

# Save figure with consistent formatting
save_figure <- function(plot_obj, filename, width = 10, height = 7, dpi = 300) {
  results_dir <- get_results_dir()
  
  # Extract subdirectory from filename if present (e.g., "km_curves/km_plot.png")
  if (grepl("/", filename)) {
    subdir <- dirname(filename)
    fig_dir <- file.path(results_dir, "figures", subdir)
    dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
    filepath <- file.path(fig_dir, basename(filename))
  } else {
    fig_dir <- file.path(results_dir, "figures")
    dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
    filepath <- file.path(fig_dir, filename)
  }
  
  # Save based on extension
  ext <- tools::file_ext(filename)
  if (ext == "png") {
    ggsave(filepath, plot = plot_obj, width = width, height = height, dpi = dpi)
  } else if (ext == "pdf") {
    ggsave(filepath, plot = plot_obj, width = width, height = height)
  } else if (ext == "svg") {
    ggsave(filepath, plot = plot_obj, width = width, height = height)
  } else {
    # Default to PNG
    filepath <- sub(paste0("\\.", ext, "$"), ".png", filepath)
    ggsave(filepath, plot = plot_obj, width = width, height = height, dpi = dpi)
  }
  
  cat("Figure saved to:", filepath, "\n")
  return(filepath)
}

# Save table to CSV
save_table <- function(table_obj, filename, subdir = NULL) {
  results_dir <- get_results_dir()
  
  if (!is.null(subdir)) {
    report_dir <- file.path(results_dir, "reports", subdir)
  } else {
    report_dir <- file.path(results_dir, "reports")
  }
  
  dir.create(report_dir, recursive = TRUE, showWarnings = FALSE)
  filepath <- file.path(report_dir, filename)
  
  write.csv(table_obj, filepath, row.names = FALSE)
  cat("Table saved to:", filepath, "\n")
  return(filepath)
}

# Format p-value for display
format_pvalue <- function(p, digits = 3) {
  if (is.na(p)) return("NA")
  if (p < 0.001) return("< 0.001")
  if (p < 0.01) return(sprintf("%.3f", p))
  if (p < 0.05) return(sprintf("%.3f", p))
  return(sprintf("%.3f", p))
}

# Format hazard ratio with CI
format_hr <- function(hr, ci_lower, ci_upper, digits = 2) {
  return(sprintf("%.2f (%.2f-%.2f)", hr, ci_lower, ci_upper))
}

# Check and install required packages
check_packages <- function() {
  required_packages <- c(
    "survival", "survminer", "ggplot2", "dplyr", "tidyr",
    "reticulate", "forestplot", "glmnet", "rms", "survivalROC"
  )
  
  missing_packages <- required_packages[!(required_packages %in% installed.packages()[,"Package"])]
  
  if (length(missing_packages) > 0) {
    cat("Missing packages:", paste(missing_packages, collapse = ", "), "\n")
    cat("Install with: install.packages(c(", 
        paste0('"', missing_packages, '"', collapse = ", "), "))\n")
    return(FALSE)
  }
  
  return(TRUE)
}
