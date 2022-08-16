# Breast-Cancer-Analysis
Using various models: Logistic regression, SVM, KNN, and Random Forest

Dataset Source: https://www.kaggle.com/datasets/reihanenamdari/breast-cancer

**About the data:**
The data contains 16 variables. They are:
- Age
- Race
- Marital Status
- T Stage
- N Stage
- 6th Stage
- differentiate
- Grade
- A Stage
- Tumor Size
- Estrogen Status
- Progesterone Status
- Regional Node Examined
- Regional Node Positive
- Survival Months
- Status

Out of these variables, the predictor variable (y) is chosen to be "Status", meaning, the status of the patients being dead or alive, depends on the other 15 independent variables. 

So, the goal is to find the best model that predicts the status (dead or alive) of a patient with highest accuracy. We will also be looking into the different metrics used to check the accuracy of the models.

**The different models used were:**
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors
- Random Forest

**Let's talk in detail about the different models used here:**
 ***1. Logistic Regression***

Logistic Regression is one of the best models for classification. The type of classification done here is binary (dead or alive), so this is a binary classification problem. The predictor variable (y) here is Status and the independent variables are the remaining 15 variables. The theory behing logistic regression is, instead of drawing a straght line to fit the datapoints, logistic regression fits a sigmoid curve by fitting the data points with minimal error. This model accuracy is 90%, meaning the model fits the data with 90% accuracy.

***2. Support Vector Machine (SVM)***

SVM algorithm draws margins to classify different groups by minimizing the errors between the data points. The accuracy of our SVM model is 90%, meaning it can predict the status of the patient with 90% accuracy.

***3. K-Nearest Neighbors (KNN)***

KNN algorithm follows a simple procedure where in a datapoint can be belonging to a cluster based on the closest n number of datapoints surrounding it. The n can be 5 or 10, and it is usually determined by trial and error, based on the dataset. The accuracy of our KNN model is 89%, meaning it can predict the status of the patient with 89% accuracy.

***4. Random Forest***

Random Forest algorithm randomly classifies the dataset and builds a decision tree for each randomly created dataset. Now, the different decision trees gives random decisions, and we take the majority vote to determine the result. The accuracy of our Random Forest model is 90%, meaning it can predict the status of the patient with 90% accuracy.

Note that, in this exercise, I have also included something called feature selection, that selects important features from the dataset. Random forest algorithm has a function called "feature_importances_" that sorts the features based on how important that feature in determining the output. When I used this, and built a model only using the important features, the accuracy of the model jumped to 100%.

**Accuracy Metrics**

We can see above that all the models above give a good accuracy of 90% and above. But does that mean they are actually good? In order to find this we can use a lot of accuracy metrics to check the accuracy of the model using various metrics, and sklearn library already has a variety of metrics to be used. Some of them used here are:
- Confusion matrix : A table that gives the model accuracy trade off between actual and predicted values
- Mean Absolute Error : gives the difference between predicted values and the actual values. The lower the value, the better the results. The measure is between 0 and 1
- Mean Squared Error : the squared error (distance) between the datapoints and the fitted model. Even here, the lower the error, the better the results. The measure is between 0 and 1
- Root Mean Squared Error : the square root of mean squared error. Even here, the lower the error, the better the results. The measure is between 0 and 1
- R2 Score : R-squared score is the variance of the dependent variable given the independent variables and usually represent how well the model has fitted the data points. It is measured between 0 and 1, and the closer it is to 1, the better the fit.

**Comparing the models for our dataset**

As seen above, all the models have high accuracy score. But given the metrics, Random forest model performs better than all the other models, even before doing the feature selection. Of course, with feature selection too, Random forest gives the best performing model in predicting the living status of the patient having breast cancer.

***Other terms used:
Hyperparameter Tuning: In order to minimize loss, and maximize the performance of the model, there are a set of hyperparameters available unique to every algorithm we use. To choose the correct hyperparameter is impossible because it is unique to every dataset we work with and depends on the size of dataset too. So usually, it is done by trial and error.***
