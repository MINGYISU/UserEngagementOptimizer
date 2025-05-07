# UserEngagementOptimizer

This project analyzes user interaction data from the **Federato RiskOps platform**, a SaaS solution used by insurance underwriters to assess risk and issue quotes. The goal is to model user behavior, identify key engagement patterns, and predict the next user action to optimize daily engagement.

## Project Overview

Using session-level event data (e.g., page visits, button clicks, form submissions), this project:

- Cleans and processes user activity logs
- Analyzes engagement trends and retention patterns
- Trains a neural network to predict the **next user action**
- Explores feature importance and behavioral signals
- Recommends strategies to improve **platform stickiness** and **workflow optimization**

## Dataset Notice

The data was obtained from **Federato RiskOps platform** and is subject to strict confidentiality agreements, making it proprietary and restricted. As such, neither data files(or visualizations) nor concrete analysis results are included in this repository. We only demostrate a high-level overview of the analysis process and coding techiques used.

## Methods and Technologies

- Python (Pandas, NumPy, Scikit-learn, PyTorch/Keras)
- Neural Network for next-action prediction
- Session path sequence modeling
- Exploratory data analysis and visualization
- Optimization for user retention

## Key Outcomes

- Learned behavioral models to guide UI/UX design improvements
- Action prediction model that helps optimize user journeys
- Insights on high-retention actions and critical drop-off points

## Future Work

- Incorporate causal inference to distinguish correlation from causation
- A/B testing integration for recommended next actions
- Dashboard for real-time session tracking and engagement analytics
