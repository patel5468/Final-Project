The answers to the questions as part of Task 4: 

###

**Question 1: How does the perturbed ensemble variance vary with time relative to the reference ensemble's variance?**

Answer: The perturbed ensemble's variance increases over time and approaches saturation relative to the reference ensemble's variance, as indicated by the normalized variance reaching approximately 1. For the perturbed ensemble, there is a small increase over time from the start date to the end date until saturation is reached. The reference ensemble increases variance immediately and stays relatively constant at each level throughout the entire time period. Level 6 exibits a change from increasing, reaching a maxima and then decreasing once again. 

###

**Question 2: For each SPEEDY model variable, for each spatial scale band and for each model level, how many days did it take for the perturbed ensemble's variance to saturate? Which spatial scale's ensemble variance saturates first? Which spatial scale's ensemble variance saturates last?**


Answer: From the normalized variance plots, we can analyze the behavior of variance saturation across spatial scales and model variables:

1. Large Spatial Scales:
   - For all variables (`u`, `v`, `t`, `q`), the variance for large spatial scales saturates relatively early, typically within 10–15 days at most levels. This is expected due to the dominant role of large-scale atmospheric processes, such as synoptic systems (i.e. troughs & ridges), which tend to stabilize quickly in ensemble forecasts. These systems operate on longer wavelengths and are less influenced by the chaotic nature of smaller-scale turbulence.
   - Among all variables, `t` (temperature) and `u` (zonal wind) exhibit faster saturation across most levels, likely due to their direct coupling with the large-scale atmospheric dynamics.

2. Medium Spatial Scales: 
   - Medium spatial scales take longer to saturate compared to large scales, typically around 15–20 days across most model levels. These scales are influenced by mesoscale phenomena, such as smaller cyclonic systems and convective complexes. Their saturation time reflects the balance between large-scale forcing and the more chaotic nature of mesoscale turbulence. Variables like `q` (specific humidity) showed a more gradual progression, with saturation taking slightly longer, particularly at mid-levels.

3. Small Spatial Scales:
   - Small-scale ensemble variances saturate last because these scales are dominated by highly localized phenomena, such as turbulence, small-scale convection, and boundary layer processes. The plots indicate that for variables like `q` and `t`, it can take 20–30 days or more for saturation to occur, especially at lower atmospheric levels. This delayed saturation reflects the slower equilibration of smaller, more chaotic features in ensemble modeling.

Overall, spatial scale-wise order of saturation:
   1. Large scales saturate first (10–15 days).  
   2. Medium scales saturate next (15–20 days).  
   3. Small scales saturate last (20–30+ days).

This pattern highlights the progressive nature of variance equilibration across scales, with smaller spatial features requiring more time to reach statistical equilibrium.
The progressive nature of variance saturation highlights the scale-dependent dynamics of the atmosphere. Larger scales, which are governed by synoptic processes, reach statistical equilibrium faster, while smaller scales, dominated by localized and turbulent processes, take longer to saturate. Variables sensitive to localized phenomena (i.e. q)  exhibited slower saturation compared to those directly linked to large-scale flow dynamics like u and t.
###
