# Problem definition (v0.1)

**Context:**  
City waste trucks currently follow static routes regardless of how full bins are, wasting fuel and time.

**Goal:**  
Use IoT bin fill data and route optimization to:
1. Forecast bin fill levels for the next 1–7 days.  
2. Select bins that must be collected the next day.  
3. Optimize truck routes to minimize total distance and time.

**Users:**  
City waste management planners and truck drivers.

**Metrics:**  
- Forecasting: MAE/MAPE for next-day fill level.  
- Routing: % reduction in total kilometers vs baseline.  
- Service quality: missed/overflowed bins per district.

**Constraints:**  
- Limited trucks, capacities, shift times.  
- Depot start/end.  
- Bins only collected if predicted > threshold (e.g. 80% full).

**Risks & Mitigation:**  
- Missing real data → synthetic dataset with same structure.  
- Scaling issues → clustering by district.  
- Fairness → track service coverage by area.
