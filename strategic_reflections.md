# Strategic Reflections

## Model Failure: my model underestimates delivery time on rainy days. Do you fix the model, the data, or the business expectations?

Fix the model. The data is accurate, we're correctly recording weather conditions and traffic levels. And customers absolutely should expect accurate ETAs regardless of weather. The problem is that our Ridge regression treats weather and traffic as additive factors, but they're actually multiplicative.

When it's clear weather with high traffic, we see about 58 minutes average delivery time. But when it's rainy with high traffic, that jumps to 72 minutes, a 14-minute difference that our linear model completely misses. Rain doesn't just add delay, it amplifies traffic delays exponentially.

The fix is straightforward: add explicit interaction features. We need features like `Weather_Traffic_Interaction` that capture these multiplicative effects. I'd also consider polynomial features to handle the non-linear relationships we're seeing. This should give us a 20-30% reduction in rainy day prediction errors, which directly translates to fewer angry customers and support tickets.

## Transferability: The model performs well in Mumbai. It's now being deployed in São Paulo. How do you ensure generalization?

My approach would be phased adaptation. Start with the Mumbai model as my foundation, it's already learned the basic physics of delivery operations. Then gradually adapt it to São Paulo's specific patterns. First, I'd add city-specific features like traffic multipliers and weather severity factors that capture São Paulo's unique conditions. 

The key is transfer learning with fine-tuning. Don't retrain from scratch, that throws away all the valuable patterns the Mumbai model learned. Instead, use smaller learning rates to gradually adapt the model to São Paulo's data while preserving the core understanding of delivery physics.

I'd also implement continuous monitoring to catch when the model starts drifting. Set up weekly performance reviews and automated alerts when prediction accuracy drops below acceptable thresholds. The goal is catching problems before customers notice them.

## GenAI Disclosure

### What parts of this project did you use GenAI for?

- Used GPT-5 Voice Mode with clear instructions for it to work as a Data Scientist during EDA, keeping constant conversation on the dataset structure, the patterns I was recognizing. Making what was practically a live simulation of pair coding and business-case discussion.

- Used Perplexity as a research assistant to find research papers on key topics such as ML Algorithms to speed up search of knowledge.

- Used Cursor both as a companion and agent. Making it check for errors, opportunities to have the code be as close to production-grade, and generate parts of the code which would usually take me more time. With Cursor agent I was clear, giving it clear plans, and it's tasks were always on code segments I had already written myself.

### How did you validate or modify it's output?

- For GPT-5, I validated it's output with my execution and comparing my advancment with previous examples of similar projects and documentation from sites like Databricks.

- For Perplextiy, I used citation tracking, and using Research mode to point Perplexity directly to ArXiv's official sources.

- For Cursor, I had it rules directly pointed to the assessment document, and all the code modified by Cursor was constantly tested, in combination with other key functionalities like constant branching to work on specific features of the project, allowing for easy rollsbacks if needed.

### Overview

I used 3 different GenAI tools at different parts of the project to have the technology work as a team for me to manage and discuss the project with.

GPT-5 was the Data Scientist and Stakeholder, helping keep track of actions, things needed to be changed, and a focus on both technical work and easy to digest information for stakeholders like the EDA Report. 

Cursor was the software team, with constant work on top of my already architected code and plans, with robust versin control and clear coding practices instructions.

Perplexity was the consultant, a perfect way to extend the the work to something deeper, allowing me to work fast, while going deep into both fundamentals and more extensive and novel topics when needed.

## Signature Insight: What's one non-obvious insight or decision you're proud of from this project?

The biggest insight was choosing physics over complexity. At first sight these scenario seems to be a task for XGBoost or neural networks, but I went the opposite direction, I focused on creating features that actually encode the physics of delivery operations.

Instead of throwing complex algorithms at raw features, I built features that capture reality: vehicle-specific speed constraints, physical time estimates, and operational reality checks. Things like `Estimated_Travel_Time = Distance × Time_per_km` that encode the fact that you can't deliver faster than physics allows.

The result? Ridge regression outperformed XGBoost. We got 82% accuracy with Ridge versus 79% with XGBoost, plus much better generalization and interpretability. Operations teams can actually understand why the model makes certain predictions, which builds trust and enables better decision-making.

This taught me that feature engineering beats algorithm selection every time. Domain knowledge is more valuable than algorithmic sophistication.

Another insight I noticed is around data types and memory efficiency. The original dataset uses float64 for most numerical features, but the actual precision in the values doesn't require that much space. Distance values like 7.93 km or preparation times like 15.0 minutes could easily be stored as float32 or even float16 without losing meaningful precision. For a small dataset like ours it doesn't matter much, but when scaling to millions of orders, downcasting could provide significant memory savings with minimal precision cost. It's the kind of optimization that becomes critical at scale but most people overlook during development.

## Going to Production: How would you deploy my model to production? What other components would you need to include/develop in my codebase?

### How would you deploy your model to production?

The deployment strategy builds on the solid foundation we already have. The FastAPI application in `api/app.py` handles the core prediction logic with health checks, model info endpoints, and both single and batch prediction capabilities. The `PredictorService` class in `api/predictor_service.py` wraps our model with proper preprocessing, handling raw input conversion and error management.

For production deployment, I'd implement a phased rollout approach. Start with shadow testing (deploy the new model alongside the current one), logging predictions without serving them to customers. This lets me compare performance on real traffic without risk. After a week of shadow testing, move to canary deployment, routing 5% of traffic to the new model while monitoring key metrics closely.

The deployment pipeline would be fully automated. Code changes trigger automated testing, model training, validation, and staged deployments. I'd implement automated rollback capabilities based on performance metrics, if prediction accuracy drops below thresholds, automatic rollback happens within minutes.

### What other components would you need to include/develop in your codebase?

**Model Versioning and Registry**
The current codebase lacks proper model versioning. I'd add a model registry that tracks every model version with metadata including training data hash, feature schema, performance metrics, and deployment status. Each model gets a unique version ID and semantic versioning scheme. The registry needs to support model promotion workflows and maintain complete audit trails.

**Feature Store Implementation**
While we have feature engineering in `model_pipeline/feature_engineering.py`, we need a production feature store. This would handle both batch and real-time features, with scheduled pipelines for historical data and on-demand computation for live features like current weather and traffic conditions. The feature store needs versioning, lineage tracking, and data quality monitoring.

**Enhanced Pipeline Monitoring**
The existing pipeline in `model_pipeline/pipeline.py` is solid but needs production enhancements. I'd add automated data validation checks at each stage - missing value detection, outlier identification, and data type validation. The pipeline needs to handle schema evolution gracefully and include comprehensive logging for debugging.

**Model Performance Monitoring**
We need continuous monitoring for data drift using statistical tests like Kolmogorov-Smirnov tests. The system should detect when input feature distributions change significantly from training data. I'd implement automated retraining triggers when performance degrades below thresholds and set up automated model validation that runs on every new model version.

**A/B Testing Framework**
The current codebase doesn't have testing frameworks. I'd implement A/B testing capabilities that can split traffic based on user IDs, geographic regions, or other criteria. The testing framework needs statistical significance testing and proper experiment isolation to avoid contamination between test groups.

**Automated Model Lifecycle**
Building on the existing `ModelTrainer` and `ModelEvaluator` classes, I'd add automated retraining pipelines that detect when model performance degrades and trigger retraining with fresh data. The system needs automated model evaluation that runs comprehensive tests including performance metrics, bias testing, and comparison against baseline models.

The existing components provide a strong foundation - the modular pipeline architecture, FastAPI application, and predictor service handle the core functionality well. The additional components focus on production reliability, monitoring, and automated model management that are essential for scaling to production workloads.
