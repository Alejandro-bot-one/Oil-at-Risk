library(tidyverse)
library(sjPlot) # Study this
library(performance) # Study this
library(quantreg) # Study this
library(readr)

# ---- Initial configuration of the data ----
theme_set(theme_bw())

# --- Data loading ----
merged_series <- read_csv(
  "~/Msc Analisis Economico Cuantitativo UAM/TFM/data/merged_series.csv",
  col_types = cols(
    Date = col_date(format = "%Y-%m-%d")
  )
)


#---- HF INDICATOR MODELS ----

merged_series$HF_indicator_12d = c(rep(NA, 12), diff(merged_series$HF_indicator, lag = 12))
  
  # 1. GPRC regressed on HF Indicator
ggplot(merged_series, aes(x = GPRC_ESP_YoY, y =  HF_indicator)) +
  geom_point(alpha = 0.5, color = "darkgray") +
  geom_smooth(method = "lm", color = "#D55E00", linewidth = 1) +
  geom_quantile(quantiles = c(0.05, 0.25, 0.5, 0.75, 0.95), color = "#0072B2", alpha = 0.75, linewidth = 1) +
  labs(title = " HF_indicator  ~ GPRC_ESP_YoY ")

# 1.2 GPRC regressed on HF Indicator 12d
ggplot(merged_series, aes(x = GPRC_ESP_YoY, y =  HF_indicator_12d)) +
  geom_point(alpha = 0.5, color = "darkgray") +
  geom_smooth(method = "lm", color = "#D55E00", linewidth = 1) +
  geom_quantile(quantiles = c(0.05, 0.25, 0.5, 0.75, 0.95), color = "#0072B2", alpha = 0.75, linewidth = 1) +
  labs(title = " HF_indicator_12d  ~ GPRC_ESP_YoY ")


# OLS (Mínimos Cuadrados Ordinarios)
m_ols_hf_gpr <- lm(HF_indicator ~ GPRC_ESP_YoY, data = merged_series)

# Quantile regressions (Regresiones Cuantílicas)
m_q05_hf_gpr <- rq(HF_indicator ~ GPRC_ESP_YoY, tau = 0.05, data = merged_series)
m_q50_hf_gpr <- rq(HF_indicator ~ GPRC_ESP_YoY, tau = 0.50, data = merged_series)
m_q95_hf_gpr <- rq(HF_indicator ~ GPRC_ESP_YoY, tau = 0.95, data = merged_series)

# Compare the models (Comparación gráfica de coeficientes)
plot_models(
  m_ols_hf_gpr, m_q05_hf_gpr, m_q50_hf_gpr, m_q95_hf_gpr,
  m.labels = c("OLS", "Q05", "Q50", "Q95"),
  title = "Coefficient Comparison: HF_indicator ~ GPRC_ESP_YoY",
  colors = c("#D55E00", "#0072B2", "#009E73", "#CC79A7") 
)

# 2. EPU regressed on HF Indicator
ggplot(merged_series, aes(x = EPU_YoY, y =  HF_indicator)) +
  geom_point(alpha = 0.5, color = "darkgray") +
  geom_smooth(method = "lm", color = "#D55E00", linewidth = 1) +
  geom_quantile(quantiles =c(0.05, 0.25, 0.5, 0.75, 0.95), color = "#0072B2", alpha = 0.75, linewidth = 1) +
  labs(title = "HF_indicator  ~ EPU_YoY") 

# OLS (Mínimos Cuadrados Ordinarios)
m_ols_hf_epu <- lm(HF_indicator ~ EPU_YoY, data = merged_series)

# Quantile regressions (Regresiones Cuantílicas)
m_q05_hf_epu <- rq(HF_indicator ~ EPU_YoY, tau = 0.05, data = merged_series)
m_q50_hf_epu <- rq(HF_indicator ~ EPU_YoY, tau = 0.50, data = merged_series)
m_q95_hf_epu <- rq(HF_indicator ~ EPU_YoY, tau = 0.95, data = merged_series)

# Compare the models (Comparación gráfica de coeficientes)
plot_models(
  m_ols_hf_epu, m_q05_hf_epu, m_q50_hf_epu, m_q95_hf_epu,
  m.labels = c("OLS", "Q05", "Q50", "Q95"),
  title = "Coefficient Comparison: HF_indicator ~ EPU_YoY",
  colors = c("#D55E00", "#0072B2", "#009E73", "#CC79A7") # Manteniendo la estética
)


# ---- TOURIST INDICATOR MODELS ----

# 3. Tourists regressed on GPRC (Geopolitical Risk)
ggplot(merged_series, aes(x = GPRC_ESP_YoY, y = N_turistas_YoY)) +
  geom_point(alpha = 0.5, color = "darkgray") +
  geom_smooth(method = "lm", color = "#D55E00", linewidth = 1) +
  geom_quantile(quantiles = c(0.05, 0.25, 0.5, 0.75, 0.95), color = "#0072B2", alpha = 0.75, linewidth = 1) +
  labs(title = "N_turistas_YoY ~ GPRC_ESP_YoY") 

# OLS (Mínimos Cuadrados Ordinarios)
m_ols_gpr <- lm(N_turistas_YoY ~ GPRC_ESP_YoY, data = merged_series)

# Quantile regressions (Regresiones Cuantílicas)
m_q05_gpr <- rq(N_turistas_YoY ~ GPRC_ESP_YoY, tau = 0.05, data = merged_series)
m_q50_gpr <- rq(N_turistas_YoY ~ GPRC_ESP_YoY, tau = 0.50, data = merged_series)
m_q95_gpr <- rq(N_turistas_YoY ~ GPRC_ESP_YoY, tau = 0.95, data = merged_series)

# Compare the models (Comparación gráfica de coeficientes)
plot_models(
  m_ols_gpr, m_q05_gpr, m_q50_gpr, m_q95_gpr,
  m.labels = c("OLS", "Q05", "Q50", "Q95"),
  title = "Coefficient Comparison: N_turistas_YoY ~ GPRC_ESP_YoY",
  colors = c("#D55E00", "#0072B2", "#009E73", "#CC79A7") # Opcional: colores diferenciados
)


# 4. Tourists regressed on EPU (Economic Policy Uncertainty)
ggplot(merged_series, aes(x = EPU_YoY, y = N_turistas_YoY)) +
  geom_point(alpha = 0.5, color = "darkgray") +
  geom_smooth(method = "lm", color = "#D55E00", linewidth = 1) +
  geom_quantile(quantiles = c(0.05, 0.25, 0.5, 0.75, 0.95), color = "#0072B2", alpha = 0.75, linewidth = 1) +
  labs(title = "N_turistas_YoY ~ EPU_YoY")

z