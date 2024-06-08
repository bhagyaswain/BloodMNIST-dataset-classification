#Log Mining and Analysis, Liability Claim Prediction, and High-Energy Physics with Ensemble Methods

##Project Description
This project encompasses several analyses including log mining and analysis, liability claim prediction, and the application of ensemble methods in high-energy physics. Each section of the project explores different datasets and employs various machine learning techniques to derive insights and predictions.

##Sections
1. Log Mining and Analysis
Description
Analyzed log data from NASA's access logs for July 1995 to determine the total number of requests from different countries, identify unique hosts, and visualize the data.

##Methodology
Tools Used: Apache Spark for processing and analyzing the log data, Matplotlib for data visualization.
Techniques: Regular expressions for extracting relevant information from log entries.
Results
Total Requests:
Germany (.de): 21,345
Canada (.ca): 58,290
Singapore (.sg): 1,057
Unique Hosts and Top Frequent Hosts:
Germany: 1,138 unique hosts
Canada: 2,970 unique hosts
Singapore: 78 unique hosts
Visualizations
Requests by Countries
Heatmap visualizations showing peaks in visitation throughout the day.
Observations
Higher number of requests from Canada compared to Germany and Singapore.
Top hosts in Canada have more requests compared to Germany and Singapore.
Clear peaks in visitation during the middle of the day and smaller peaks in the evening.
2. Liability Claim Prediction
Description
Predicted the likelihood and number of liability claims using Poisson and Logistic regression models.

##Methodology
Data Preprocessing: Converted dataset to PySpark DataFrame and created new columns indicating the presence of claims.
Model Training: Used Poisson regression and Logistic regression with cross-validation to find optimal parameters.
Results
Poisson Regression: RMSE of 0.248
Logistic Regression:
L1 Regularization: AUC of 0.5
L2 Regularization: AUC of 0.629, accuracy of 0.949
3. High-Energy Physics using Ensemble Methods
Description
Used supervised classification algorithms to identify Higgs bosons from particle collisions using the HIGGS dataset.

##Methodology
Algorithms Used: Random Forests, Gradient Boosting, and Neural Networks.
Parameter Tuning: Employed pipelines and cross-validation on a subset of the data.
Results
AUC on Full Dataset:
Random Forest: 0.688
Gradient Boosting: 0.723
Multilayer Perceptron: 0.623
Discussion
Gradient Boosting achieved the highest AUC, indicating superior performance in identifying Higgs bosons.

4. Movie Recommendation and Cluster Analysis
Description
Performed time-split recommendation and user clustering using the ALS algorithm on movie ratings data.

##Methodology
ALS Settings: Experimented with different ALS parameters to improve recommendation accuracy.
Metrics: RMSE, MSE, and MAE were computed for different splits and ALS settings.
Results
Top Movies and Genres: Identified top-rated movies and popular genres within the largest user cluster for each split.
User Clustering: Analyzed clusters of users and the top five user clusters for each data split.
Observations
RMSE, MSE, and MAE values decrease as the size of the training data increases.
Shifts in the popularity of genres over different time periods.
