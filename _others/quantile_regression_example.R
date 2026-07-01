library(tidyverse)
library(sjPlot) # Study this
library(performance) # Study this
library(quantreg) # Study this

# ---- Initial configuration of the data ----
theme_set(theme_bw())
data(engel)

# Visualize the original data
ggplot(engel, aes(foodexp,income))+
  geom_point()

# Taking logs of the income and food expenditure, we observe the problem
# is not solved at all.

engel$log_income = log(engel$income)
engel$log_foodexp = log(engel$foodexp)

ggplot(engel, aes(log_foodexp,log_income))+
  geom_point()+
  geom_smooth(method = lm, se = F, color = "red")

# Performance of the linear regression

lin_reg = lm(log_foodexp ~ log_income, data = engel)

summary(lin_reg) # Normal linear reg stats
check_heteroskedasticity(lin_reg)

# Now lets try a quantile regression
ggplot(engel, aes(log_foodexp,log_income))+
  geom_point()+
  geom_smooth(method = lm, se = F, color = "red")+
  geom_quantile(color = "blue", alpha = .5 ,quantiles = c(.25,.5,.75))
