python version 3.13.13

Sergei Patrushev

Project: Student Math Performance Analysis and Grade Prediction


This project explores the Student Performance dataset and builds linear regression models to predict final math grades (G3). The analysis includes data cleaning, exploratory data analysis (EDA), feature engineering, visualization, and model evaluation.

Project Goals
Explore factors that influence student academic performance.
Investigate relationships between attendance, study habits, family background, and grades.
Build and evaluate regression models for grade prediction.
Compare a simple baseline model against a more comprehensive model.
Examine the predictive impact of previous grades (G1).
Key Tasks
Data loading and exploration
Handling students who did not take the final exam (G3 = 0)
Correlation analysis
Data visualization
Linear regression modeling
Model evaluation using RMSE and R²
Predicted vs. actual performance analysis
Results

The baseline model using only past failures showed limited predictive power. Adding additional demographic and academic features improved performance modestly. Including the first-period grade (G1) dramatically increased predictive accuracy, highlighting the strong relationship between earlier and final course performance.

Technologies Used
Python
Pandas
NumPy
Matplotlib
Scikit-learn
Outputs

The project generates several visualizations, including:

Distribution of final grades
Absences vs. final grade analysis
Feature relationship plots
Predicted vs. actual grade comparison

These outputs are saved in the outputs/ directory.