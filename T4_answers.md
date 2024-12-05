The answers to the questions as part of Task 4: 

###

**Question 1: How does the perturbed ensemble variance vary with time relative to the reference ensemble's variance?**

Answer: The perturbed ensemble's variance increases over time and approaches saturation relative to the reference ensemble's variance, as indicated by the normalized variance reaching approximately 1. For the perturbed ensemble, there is a small increase over time from the start date to the end date until saturation is reached. The reference ensemble increases variance immediately and stays relatively constant at each level throughout the entire time period. Level 6 exibits a change from increasing, reaching a maxima and then decreasing once again. 

###

**Question 2: For each SPEEDY model variable, for each spatial scale band and for each model level, how many days did it take for the perturbed ensemble's variance to saturate? Which spatial scale's ensemble variance saturates first? Which spatial scale's ensemble variance saturates last?**


Answer: From the normalized variance plots:

1. Large Spatial Scales:
   - For all variables (`u`, `v`, `t`, `q`), the variance for large spatial scales saturates relatively early, typically within 10–15 days at most levels. This is expected since large-scale features like synoptic systems tend to equilibrate faster in ensemble forecasting.  
   - Among all variables, `t` (temperature) and `u` (zonal wind) exhibit faster saturation across most levels.

2. Medium Spatial Scales: 
   - Medium spatial scales take longer to saturate compared to large scales, typically around 15–20 days across most model levels. Variables like `q` (specific humidity) show a more gradual progression, with saturation taking slightly longer, particularly at mid-levels. 

3. Small Spatial Scales:
   - Small-scale ensemble variances saturate last. The plots indicate that for variables like `q` and `t`, it can take 20–30 days or more for saturation to occur, especially at lower atmospheric levels. This delayed saturation reflects the slower equilibration of smaller, more chaotic features in ensemble modeling.

Overall, spatial scale-wise order of saturation:
   1. Large scales saturate first (10–15 days).  
   2. Medium scales saturate next (15–20 days).  
   3. Small scales saturate last (20–30+ days).

This pattern highlights the progressive nature of variance equilibration across scales, with smaller spatial features requiring more time to reach statistical equilibrium.

###
