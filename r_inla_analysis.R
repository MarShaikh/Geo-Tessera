#!/usr/bin/env Rscript
#
# R-INLA Analysis with GeoTessera Embeddings
#
# This script demonstrates how to use the extracted embedding features
# in spatial models with R-INLA

library(INLA)
library(sf)
library(tidyverse)
library(spdep)

# ============================================================================
# 1. Load Data
# ============================================================================

# Load extracted embedding features
# Choose LGA or State level
embeddings_lga <- read_csv("nigeria_embeddings/processed/nigeria_embeddings_lga_all_years.csv")
embeddings_state <- read_csv("nigeria_embeddings/processed/nigeria_embeddings_state_all_years.csv")

# For this example, use state-level data
data <- embeddings_state

# Load spatial boundaries
state_boundaries <- st_read("nigeria_states.geojson")

cat("Loaded data:\n")
cat(sprintf("  %d state-year observations\n", nrow(data)))
cat(sprintf("  %d unique states\n", length(unique(data$state_id))))
cat(sprintf("  %d unique years\n", length(unique(data$year))))
cat(sprintf("  %d embedding features per observation\n",
            sum(str_starts(names(data), "emb_"))))

# ============================================================================
# 2. Dimensionality Reduction (PCA)
# ============================================================================

# Extract embedding columns
emb_cols <- names(data)[str_starts(names(data), "emb_mean_")]
cat(sprintf("\nReducing %d embedding dimensions...\n", length(emb_cols)))

# Perform PCA on mean embeddings only
emb_matrix <- data %>%
  select(all_of(emb_cols)) %>%
  as.matrix()

# Handle missing values
emb_matrix[is.na(emb_matrix)] <- 0

# PCA
pca_result <- prcomp(emb_matrix, scale. = TRUE, center = TRUE)

# Determine number of components (e.g., 90% variance explained)
variance_explained <- cumsum(pca_result$sdev^2) / sum(pca_result$sdev^2)
n_components <- which(variance_explained >= 0.90)[1]

cat(sprintf("  Using %d components (%.1f%% variance explained)\n",
            n_components, variance_explained[n_components] * 100))

# Add PCA components to data
pca_df <- as.data.frame(pca_result$x[, 1:n_components])
names(pca_df) <- paste0("PC", 1:n_components)

data <- bind_cols(data, pca_df)

# ============================================================================
# 3. Prepare Spatial Structure for INLA
# ============================================================================

# Create neighborhood structure based on adjacency
# This assumes your state boundaries have proper topology

# Match state IDs between data and boundaries
state_boundaries <- state_boundaries %>%
  arrange(state_id)

# Create adjacency matrix
nb <- poly2nb(state_boundaries, queen = TRUE)

# Save as INLA graph file
nb2INLA("nigeria_state.adj", nb)
g <- inla.read.graph("nigeria_state.adj")

cat(sprintf("\nSpatial structure: %d states, %d neighbors on average\n",
            length(nb), mean(sapply(nb, length))))

# ============================================================================
# 4. Example: Disease Mapping Model
# ============================================================================

# Example outcome: simulate disease cases (replace with your real data)
set.seed(123)
data <- data %>%
  mutate(
    # Simulate population
    population = rpois(n(), lambda = 100000),
    # Simulate disease cases (replace with your actual data!)
    cases = rpois(n(), lambda = population * 0.001 * (1 + 0.1 * PC1))
  )

# Create state ID for spatial random effect
data <- data %>%
  group_by(state_id) %>%
  mutate(area_idx = cur_group_id()) %>%
  ungroup()

# ============================================================================
# Model 1: Simple model with top PCs and spatial effect
# ============================================================================

cat("\n" * "="*60)
cat("Model 1: Poisson model with PCs and spatial random effect\n")
cat("="*60, "\n")

formula1 <- cases ~
  offset(log(population)) +
  PC1 + PC2 + PC3 + PC4 + PC5 +  # Top 5 PCs as covariates
  f(area_idx, model = "besag", graph = g)  # Spatial random effect

fit1 <- inla(
  formula1,
  family = "poisson",
  data = data,
  control.predictor = list(compute = TRUE),
  control.compute = list(
    dic = TRUE,
    waic = TRUE,
    cpo = TRUE,
    config = TRUE
  ),
  control.inla = list(strategy = "adaptive"),
  verbose = FALSE
)

summary(fit1)

# Fixed effects
cat("\nFixed effects (PCs):\n")
print(fit1$summary.fixed)

# Model fit
cat("\nModel fit:\n")
cat(sprintf("  DIC: %.2f\n", fit1$dic$dic))
cat(sprintf("  WAIC: %.2f\n", fit1$waic$waic))

# ============================================================================
# Model 2: Spatio-temporal model (if you have multiple years)
# ============================================================================

if (length(unique(data$year)) > 1) {
  cat("\n", "="*60)
  cat("Model 2: Spatio-temporal model\n")
  cat("="*60, "\n")

  # Create time index
  data <- data %>%
    mutate(time_idx = as.numeric(factor(year)))

  n_areas <- length(unique(data$area_idx))
  n_times <- length(unique(data$time_idx))

  # Create space-time index
  data <- data %>%
    mutate(st_idx = area_idx + n_areas * (time_idx - 1))

  formula2 <- cases ~
    offset(log(population)) +
    PC1 + PC2 + PC3 + PC4 + PC5 +
    f(area_idx, model = "besag", graph = g) +  # Spatial effect
    f(time_idx, model = "rw1") +  # Temporal trend
    f(st_idx, model = "iid")  # Space-time interaction

  fit2 <- inla(
    formula2,
    family = "poisson",
    data = data,
    control.predictor = list(compute = TRUE),
    control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
    control.inla = list(strategy = "adaptive"),
    verbose = FALSE
  )

  summary(fit2)

  cat("\nModel comparison:\n")
  cat(sprintf("  Model 1 (spatial only) WAIC: %.2f\n", fit1$waic$waic))
  cat(sprintf("  Model 2 (spatio-temporal) WAIC: %.2f\n", fit2$waic$waic))
}

# ============================================================================
# Model 3: Variable selection with all PCs
# ============================================================================

cat("\n", "="*60)
cat("Model 3: Variable selection for all PCs\n")
cat("="*60, "\n")

# Build formula with all PCs
pc_names <- paste0("PC", 1:n_components)
pc_formula <- paste(pc_names, collapse = " + ")

formula3 <- as.formula(paste0(
  "cases ~ offset(log(population)) + ",
  pc_formula,
  " + f(area_idx, model = 'besag', graph = g)"
))

# Use penalized complexity priors for regularization
fit3 <- inla(
  formula3,
  family = "poisson",
  data = data,
  control.predictor = list(compute = TRUE),
  control.compute = list(dic = TRUE, waic = TRUE),
  control.fixed = list(
    # This adds L2 penalty (ridge-like)
    prec = list(default = 0.001)
  ),
  verbose = FALSE
)

# Identify significant PCs (95% credible interval doesn't include 0)
sig_pcs <- fit3$summary.fixed %>%
  as.data.frame() %>%
  rownames_to_column("variable") %>%
  filter(str_starts(variable, "PC")) %>%
  filter(`0.025quant` * `0.975quant` > 0) %>%  # Same sign for CI bounds
  pull(variable)

cat(sprintf("\nSignificant PCs (%d out of %d):\n",
            length(sig_pcs), n_components))
print(sig_pcs)

# ============================================================================
# 5. Visualization
# ============================================================================

# Add predictions to spatial data
predictions <- data %>%
  mutate(
    expected = population * sum(cases) / sum(population),
    SMR = cases / expected,  # Standardized morbidity ratio
    fitted = fit1$summary.fitted.values$mean,
    fitted_rate = fitted / population * 1000,  # per 1000
    lower = fit1$summary.fitted.values$`0.025quant`,
    upper = fit1$summary.fitted.values$`0.975quant`
  )

# Join with spatial boundaries for mapping
spatial_predictions <- state_boundaries %>%
  left_join(predictions, by = "state_id")

# Plot: Raw SMR vs Fitted rates
library(ggplot2)

p1 <- ggplot(spatial_predictions %>% filter(year == max(year))) +
  geom_sf(aes(fill = SMR)) +
  scale_fill_viridis_c(option = "plasma") +
  labs(title = "Observed SMR (most recent year)") +
  theme_minimal()

p2 <- ggplot(spatial_predictions %>% filter(year == max(year))) +
  geom_sf(aes(fill = fitted_rate)) +
  scale_fill_viridis_c(option = "plasma") +
  labs(title = "Fitted rate (per 1000, most recent year)") +
  theme_minimal()

# Save plots
ggsave("nigeria_smr_map.png", p1, width = 10, height = 8)
ggsave("nigeria_fitted_map.png", p2, width = 10, height = 8)

cat("\n✓ Saved maps: nigeria_smr_map.png, nigeria_fitted_map.png\n")

# ============================================================================
# 6. Save Results
# ============================================================================

# Save predictions
write_csv(predictions, "nigeria_inla_predictions.csv")

# Save model summaries
sink("nigeria_inla_model_summary.txt")
cat("="*60, "\n")
cat("R-INLA Model Summary\n")
cat("="*60, "\n\n")
summary(fit1)
sink()

cat("\n✓ Analysis complete!\n")
cat("  - Predictions: nigeria_inla_predictions.csv\n")
cat("  - Model summary: nigeria_inla_model_summary.txt\n")
cat("  - Maps: nigeria_smr_map.png, nigeria_fitted_map.png\n")
