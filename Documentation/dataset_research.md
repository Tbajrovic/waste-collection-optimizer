# Dataset Research Report  
**Project:** Waste Collection Optimization (IoT + Route Optimization)  
**Author:** Tarik Bajrović  
**Date:** November 2025  

---

## 1. Purpose  
The goal of this research was to identify publicly available datasets relevant to smart waste management and route optimization. The ideal dataset would include information about bin locations, waste amounts or fill levels over time, and optionally truck routes or collection frequencies.  

When such data was not available in one place, partial datasets were also considered to simulate missing variables.

---

## 2. Datasets Reviewed  

| # | Dataset Name | Source / Platform | Type | Description | Pros | Cons |
|---|---------------|------------------|-------|--------------|------|------|
| **1** | **Gold Coast City – Waste Collection Points** | Kaggle / data.gov.au | Geospatial (bin locations) | Geo-layer of waste collection points for Gold Coast City (Australia). | ✅ Real municipal data; includes coordinates of bins and areas. | ❌ No time series or fill level data. |
| **2** | **Plastic Bottle Waste (Wastebase)** | Kaggle / Wastebase platform | Crowdsourced time-series | Reports of single-use plastic waste across multiple countries (2021–2024). | ✅ Temporal component; shows waste quantity trends. | ❌ No truck or bin-level detail; only aggregated waste counts. |
| **3** | **Solid Waste and Recycling Collection Routes – Town of Cary (USA)** | Kaggle / Town of Cary Open Data | Geospatial + route-level | GIS data of recycling and garbage collection areas and schedules. | ✅ Real collection routes; usable for routing optimization. | ❌ No actual waste volume or bin-level information. |
| **4** | **Singapore Waste Management** | Kaggle / NEA Singapore | Statistical / time-series | National-level recycling and energy recovery statistics (2003–2020). | ✅ Long time range; annual waste and recycling trends. | ❌ Aggregated by year; lacks spatial or bin detail. |
| **5** | **Municipal Waste and Reduction of Biodegradable Waste (Cyprus)** | data.europa.eu (Cyprus Open Data) | National statistics | Official government dataset on municipal waste quantities and biodegradable waste. | ✅ Reliable and official; part of EU reporting. | ❌ Missing timestamps and spatial resolution. |
| **6** | **Waste and Resources – Municipal Waste (Luxembourg)** | data.europa.eu (Luxembourg Open Data) | Statistical / annual | Annual reporting of municipal waste from 2020–2025. | ✅ Regularly updated, multiple years. | ❌ Only yearly data; unsuitable for short-term prediction. |
| **7** | **What a Waste – Global Database (World Bank)** | World Bank Data 360 | Global aggregated | Global dataset covering waste generation, composition, and collection for 330+ cities. | ✅ Widely used, rich metadata, global coverage. | ❌ No local detail; not suitable for route simulation. |

---

## 3. Evaluation Criteria  
| Criterion | Explanation | Priority |
|------------|--------------|----------|
| **Temporal Resolution** | Includes daily or weekly data for forecasting bin fill levels. | ⭐⭐⭐⭐ |
| **Spatial Resolution** | Contains location or route information for mapping bins or areas. | ⭐⭐⭐⭐ |
| **Data Completeness** | Few missing values and clear metadata. | ⭐⭐⭐ |
| **Practical Relevance** | Aligns with project tasks (IoT prediction + CVRP routing). | ⭐⭐⭐⭐ |
| **License & Accessibility** | Freely available and open for academic use. | ⭐⭐⭐⭐ |

---

## 4. Selected Datasets for Use  

### **Primary Choice:**  
**Solid Waste and Recycling Collection Routes (Town of Cary, USA)**  
- Provides real-world collection routes, areas, and schedules.  
- Perfect for modeling **routing optimization** using OR-Tools.  
- Will be complemented by synthetic bin-level time-series (for forecasting).  

### **Secondary (Supplementary) Data:**  
- **Gold Coast City Waste Points** → for bin-level geospatial context.  
- **Plastic Bottle Waste** → for realistic waste quantity/time pattern simulation.

---

## 5. Integration Plan  
To combine real and synthetic elements:
1. Use **Town of Cary** dataset for route structures and area definitions.  
2. Add **synthetic bin-level time-series** using the generator developed earlier.  
3. Validate forecasting and routing algorithms on these realistic spatial patterns.  

---

## 6. References  
- Kaggle Datasets:  
  - [Gold Coast City – Waste Collection Points](https://www.kaggle.com/datasets/mariliaprata/gold-coast-city-waste-collection-points)  
  - [Plastic Bottle Waste](https://www.kaggle.com/datasets/wastebase/plastic-bottle-waste)  
  - [Solid Waste and Recycling Collection Routes](https://www.kaggle.com/datasets/dariachemkaeva/solid-waste-and-recycling-collection-routes)  
  - [Singapore Waste Management](https://www.kaggle.com/datasets/abidaliwan/singapore-waste-management)  
- [data.europa.eu](https://data.europa.eu) – Luxembourg & Cyprus Waste Datasets  
- [World Bank – What a Waste Global Database](https://datacatalog.worldbank.org/search/dataset/0039597/what-a-waste-global-database)
