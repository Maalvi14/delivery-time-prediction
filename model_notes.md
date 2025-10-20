# Model Development Notes

## Executive Summary

### Context
After the EDA revealed clear patterns in delivery times, we needed to build a production-ready model that could predict ETAs accurately enough for operational decisions. The goal was to create something that Operations could trust for dispatch decisions and customers could rely on for timing expectations.

### What we learned
- `Simple models work best.` Ridge regression outperformed complex tree ensembles—sometimes less is more.
- `Feature engineering matters more than algorithm choice.` The domain features we created had bigger impact than switching between XGBoost and Random Forest.
- `Distance is still king.` Even with all our fancy features, trip distance explains most of the variance.
- `Weather and traffic are predictable delays.` These add consistent minutes that we can model reliably.
- `Overfitting is real.` Tree models looked great on training data but didn't generalize as well.

### What we can deploy
- Ridge regression model that predicts delivery times within ~6 minutes on average
- Confidence intervals for each prediction to flag high-risk orders
- Feature importance rankings to guide operational improvements
- Modular pipeline that can be retrained weekly without breaking production

### What Engineering can build
- **Real-time prediction API** that takes order details and returns ETA + confidence
- **Batch prediction service** for historical analysis and A/B testing
- **Model monitoring dashboard** to track performance drift over time
- **Automated retraining pipeline** that updates models when performance degrades

---

## Data Preparation & Preprocessing

### Missing Data Strategy
We had 3% missing values across 4 columns, not terrible, but needed careful handling. Used mode imputation for categoricals (Weather, Traffic_Level, Time_of_Day) and median for numericals (Courier_Experience_yrs). This preserves the data distribution while avoiding the bias that mean imputation can introduce.

The key insight: fit the imputation on training data only to prevent data leakage. This is a common mistake that can inflate performance metrics.

### Outlier Handling
Only 0.6% of delivery times were outliers (above 116 minutes). Rather than dropping them, we used winsorization-capping extreme values at the 1.5×IQR bounds. These long deliveries represent real operational scenarios (bad weather + long distance + traffic), so the model needs to see them.

### Feature Encoding
One-hot encoded all categorical variables but dropped the first category to avoid multicollinearity. This is standard practice for linear models. Also converted categoricals to pandas category type for memory efficiency—small optimization but good practice.

## Feature Engineering

### Domain Features
Created features based on delivery operations knowledge:

- **Estimated_Speed_kmh**: Bike (15 km/h), Scooter (25 km/h), Car (30 km/h) — realistic city speeds
- **Time_per_km**: Minutes needed per kilometer based on vehicle type
- **Estimated_Travel_Time**: Pure travel time = Distance × Time_per_km
- **Total_Time_Estimate**: Baseline estimate = Preparation + Travel time

These features help the model understand the physics of delivery—you can't deliver faster than the vehicle allows.

### Binary Indicators
Simplified complex categoricals into actionable flags:

- **Is_Rush_Hour**: Morning/Evening periods when traffic is worse
- **Is_Bad_Weather**: Rainy, Snowy, Foggy conditions that slow delivery
- **Is_High_Traffic**: High traffic level flag

Binary features are easier for linear models to learn and more interpretable for business users.

### Categorical Binning
Grouped continuous variables into meaningful categories:

- **Experience_Level**: Novice (0-2 yrs), Intermediate (2-5 yrs), Expert (5+ yrs)
- **Distance_Category**: Very_Short (0-5 km), Short (5-10 km), Medium (10-15 km), Long (15+ km)

Binning reduces noise and makes patterns clearer. Also helps with non-linear relationships.

### Interaction Features
Created combinations that matter operationally:

- **Weather_Traffic**: How weather interacts with traffic conditions
- **Vehicle_Traffic**: How vehicle type performs in different traffic

These capture the reality that bad weather + high traffic = much longer delivery times.

**Result**: 9 original features → 20 engineered features. The domain features had the biggest impact on model performance.

## Model Selection & Training

### Why Test Multiple Models?
Learned this the hard way over the years—you never know which algorithm will work best until you try them. Different models have different strengths:

- **Linear models**: Fast, interpretable, work well when relationships are roughly linear
- **Tree models**: Handle non-linear patterns, interactions, but can overfit easily  
- **Ensemble methods**: Often more accurate but harder to interpret and deploy

Started with a broad test to see what the data responds to, then focused on the promising ones.

### Model Portfolio & Reasoning
Tested 12 different algorithms with specific reasons for each:

**Linear Models** (fast, interpretable):
- **Linear Regression**: Baseline—if this doesn't work, we have bigger problems
- **Ridge Regression**: Adds regularization to prevent overfitting, handles multicollinearity
- **Lasso Regression**: Feature selection built-in, good for sparse solutions
- **ElasticNet**: Combines Ridge + Lasso, balances regularization approaches

**Tree Models** (non-linear patterns):
- **Decision Tree**: Simple baseline for tree methods, easy to interpret
- **Random Forest**: Reduces overfitting through bagging, handles missing data well
- **Gradient Boosting**: Often more accurate than Random Forest, good for structured data
- **AdaBoost**: Different boosting approach, sometimes works when others don't
- **XGBoost**: Industry standard for tabular data, usually performs well
- **LightGBM**: Faster than XGBoost, good for large datasets

**Other Approaches**:
- **K-Nearest Neighbors**: Non-parametric, good baseline for local patterns
- **Support Vector Regression**: Can handle non-linear relationships with kernels

Started with default hyperparameters to get a baseline, then tuned the promising ones.

### Hyperparameter Strategy
Used sensible defaults based on experience, didn't go crazy with tuning:

- **Ridge/Lasso**: alpha=1.0 (moderate regularization—not too weak, not too strong)
- **Random Forest**: 100 trees, max_depth=10 (enough trees for stability, shallow enough to prevent overfitting)
- **XGBoost/LightGBM**: 100 estimators, max_depth=5 (conservative settings—these models can overfit quickly)

The philosophy: get a working baseline first, then optimize if needed. Most of the time, good defaults + good features beat hyperparameter tuning + mediocre features.

Didn't do extensive hyperparameter tuning, as we wanted to see which algorithms were worth the effort first.

### Training Process
Split data 80/20 train/test, used 5-fold cross-validation for robustness. Scaled all features with StandardScaler which was essential for linear models, helpful for others.

**Why 80/20 split?** With 1000 samples, 200 test samples give us enough statistical power to trust the results. 5-fold CV gives us 5 different train/test splits to check stability.

**Why StandardScaler?** Linear models are sensitive to feature scales. Distance_km ranges 0.6-20, while Preparation_Time_min ranges 5-29. Without scaling, Distance would dominate. Tree models don't need scaling, but it doesn't hurt.

The key insight: tree models looked amazing on training data but didn't generalize as well. Classic overfitting.

## Model Performance

### Why These Metrics Matter
Picked metrics that Operations and customers actually care about:

- **R²**: Shows how much of the delivery time variation we can actually predict. 0.82 means we explain 82% of the variancethat's solid for a business problem. Operations needs to know if the model is reliable enough to trust.

- **MAE**: Average error in minutes. This is what customers feel—if we're off by 6 minutes on average, that's acceptable. RMSE penalizes big mistakes more, but MAE is more intuitive for business stakeholders.

- **RMSE**: Root mean square error. Useful because it penalizes really bad predictions (like being off by 30+ minutes). One terrible prediction can ruin customer experience, so we need to track this.

- **MAPE**: Percentage error. Helps compare performance across different delivery time ranges. A 10-minute error on a 20-minute delivery (50% error) is much worse than on a 60-minute delivery (17% error).

R² is our primary metric because it tells us if the model is fundamentally working. The others help us understand the practical impact.

### Cross-Validation Results
5-fold CV gives us confidence in the results:

- **Ridge Regression**: CV R² = 0.746 (±0.052) — most stable
- **Linear Regression**: CV R² = 0.746 (±0.052) — identical performance
- **Lasso Regression**: CV R² = 0.743 (±0.043) — good stability
- **LightGBM**: CV R² = 0.711 (±0.048) — moderate stability
- **Random Forest**: CV R² = 0.694 (±0.045) — lower stability

The linear models are more stable-less variance across folds.

### Best Model: Ridge Regression
Surprising winner with the best balance of performance and stability:

- **Test R²**: 0.820 (82% variance explained)
- **Test RMSE**: 8.98 minutes
- **Test MAE**: 6.04 minutes  
- **Test MAPE**: 10.77%

Ridge slightly outperformed plain Linear Regression, suggesting some regularization helps.

### Performance Ranking (Top 5)
1. **Ridge Regression**: R² = 0.820, RMSE = 8.98 min
2. **Linear Regression**: R² = 0.819, RMSE = 9.00 min  
3. **Lasso Regression**: R² = 0.803, RMSE = 9.35 min
4. **LightGBM**: R² = 0.790, RMSE = 9.65 min
5. **Random Forest**: R² = 0.786, RMSE = 9.72 min

Linear models dominated, showing that sometimes simple is better.

### Overfitting Analysis
Checked train vs test performance to catch overfitting:

- **Ridge Regression**: Gap = 0.0006 — excellent generalization
- **Linear Regression**: Gap = 0.0006 — identical performance  
- **Lasso Regression**: Gap = 0.016 — slight overfitting
- **Tree models**: Gap = 0.05-0.15 — moderate overfitting
- **Ensemble methods**: Gap = 0.10+ — significant overfitting

Rule of thumb: gap > 0.1 means overfitting. Ridge and Linear Regression passed with flying colors.

### Feature Importance
From the tree models (Ridge doesn't have feature importance):

1. **Distance_km**: Still the primary driver (correlation = 0.781)
2. **Preparation_Time_min**: Secondary factor (correlation = 0.307)  
3. **Weather conditions**: Significant impact on delivery times
4. **Traffic_Level**: High traffic adds predictable delays
5. **Courier_Experience_yrs**: Weak negative correlation (-0.089)

Distance dominates, but weather and traffic matter more than I expected. Experience has surprisingly little impact.

### Residual Analysis
Looked at prediction errors to understand model behavior:

- **Mean residual**: -0.67 minutes (slight underprediction bias)
- **Residual std**: 8.98 minutes
- **Distribution**: Right-skewed (non-normal)
- **Worst cases**: Max overprediction = 50 min, Max underprediction = -19 min

The model slightly underestimates delivery times on average. The right skew suggests it struggles more with very long deliveries.

---