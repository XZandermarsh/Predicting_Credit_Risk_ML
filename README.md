# Predicting_Credit_Risk_ML
## Overview
* The purpose of this analysis was to compare different machine learning models in their ability to predict credit risk based on a given data set. Credit data is inherently imbalanced because the number of "low risk" loans far outweighs the "high risk" loans. This means our data needs to be modified in some way to balance the data in each category. The types used for this analysis are Naive Random Oversampling, SMOTE Oversampling, Cluster Centroid Undersampling, SMOTEENN algorithm, Balanced Random Forest Classifier, and Easy Ensemble AdaBoost Classifier.

## Results
* Initial data contained 68470 low risk entries, and only 347 high risk entries.

![alt text](https://github.com/XZandermarsh/Predicting_Credit_Risk_ML/blob/main/Oversampling_pre.png "Original Data Breakdown")

### Naive Random Oversampling
* After resampling our data to the Naive Random Oversampling model, we have the same number of low risk and high risk samples (51366) to train the Logistic Regression model.

![alt text](https://github.com/XZandermarsh/Predicting_Credit_Risk_ML/blob/main/Oversampling_post.png "Naive Random Oversampling Count")

* The balanced accuracy score was 0.625, with an f1 score of .02 for high risk and 0.75 for low risk, for an avg/total of 0.75.

![alt text](https://github.com/XZandermarsh/Predicting_Credit_Risk_ML/blob/main/Oversampling_BAS_CRI.png "Naive Random Oversampling Scores")

### SMOTE Oversampling
* After resampling our data to the Naive Random Oversampling model, we have the same number of low risk and high risk samples (68470) to train the Logistic Regression model.

![alt text](https://github.com/XZandermarsh/Predicting_Credit_Risk_ML/blob/main/SMOTE_Oversampling_post.png "SMOTE Oversampling Count")

* The balanced accuracy score was 0.668, with an f1 score of .02 for high risk and 0.77 for low risk, for an avg/total of 0.77, a slight improvement over Naive Random Oversampling.

![alt text](https://github.com/XZandermarsh/Predicting_Credit_Risk_ML/blob/main/SMOTE_Oversampling_BAS_CRI.png "SMOTE Oversampling Scores")

### Cluster Centroid Undersampling
* After resampling our data to the Naive Random Oversampling model, we have the same number of low risk and high risk samples (246) to train the Logistic Regression model. 

![alt text](https://github.com/XZandermarsh/Predicting_Credit_Risk_ML/blob/main/Undersampling_post.png "Cluster Centroid Undersampling Count")

* The balanced accuracy score was 0.550, with an f1 score of .01 for high risk and 0.58 for low risk, for an avg/total of 0.57, worse than either oversampling result.

![alt text](https://github.com/XZandermarsh/Predicting_Credit_Risk_ML/blob/main/Undersampling_BAS_CRI.png "Cluster Centroid Undersampling Scores")

### SMOTEENN Algorithm
* After resampling our data to the Naive Random Oversampling model, we have a similar number of low risk and high risk samples to train the Logistic Regression model, but not exact ('high_risk': 68458, 'low_risk': 62022). 

![alt text](https://github.com/XZandermarsh/Predicting_Credit_Risk_ML/blob/main/SMOTEENN_post.png "SMOTEENN Count")

* The balanced accuracy score was 0.668, with an f1 score of .02 for high risk and 0.70 for low risk, for an avg/total of 0.69.

![alt text](https://github.com/XZandermarsh/Predicting_Credit_Risk_ML/blob/main/SMOTEENN_BAS_CRI.png "SMOTEENN Scores")

### Balanced Random Forest Classifier
* After resampling and fitting our data to the Naive Random Oversampling model, we have the following results. The balanced accuracy score was 0.717, with an f1 score of .11 for high risk and 0.98 for low risk, for an avg/total of 0.97. 

![alt text](https://github.com/XZandermarsh/Predicting_Credit_Risk_ML/blob/main/Balanced_Random_Forest_Classifier_BAS_CRI.png "Balanced Random Forest Classifier Scores")

The results also show the top 5 most important features were loan amount, interest rate, installment, annual income, and dti.

![alt text](https://github.com/XZandermarsh/Predicting_Credit_Risk_ML/blob/main/Balanced_Random_Forest_Classifier_feature_importance.png "Balanced Random Forest Classifier Features")

### Easy Ensemble AdaBoost Classifier
* After resampling our data to the Naive Random Oversampling model, we have the same number of low risk and high risk samples to train the Logistic Regression model. The balanced accuracy score was 0.721, with an f1 score of .03 for high risk and 0.86 for low risk, for an avg/total of 0.85.

![alt text](https://github.com/XZandermarsh/Predicting_Credit_Risk_ML/blob/main/Easy_Ensemble_AdaBoost_Classifier_BAS_CRI.png "Easy Ensemble AdaBoost Classifier Scores")


## Summary/Conclusions
* The worst performing model attempted was the Cluster Centroid Undersampling Algorithm. This may be due to the fact that there was a relatively small sample size to begin with for the high risk class. Therefore, removing a large portion of the dataset does not improve the accuracy of the model.
* Naive Random Oversampling and SMOTE Oversampling provided very similar results, although SMOTE had slightly better scores.
* The combination algorithm in SMOTEENN provided the same balanced accuracy score as SMOTE Oversampling alone, but with a worse f1 score, so it would not be recommended for this dataset.
* Both of the Ensemble models provided significantly higher balanced accuracy and f1 scores vs the resampling + logistic regression models. Of the two Ensemble models, the Easy Ensemble AdaBoost Classifier yielded the higher balanced accuracy score, but only slightly. However, the Balanced Random Forest Classifier had an f1 for high_risk that was significantly higher than any of the other algorithms attempted. For this reason, the Balanced Random Forest Classifier would be recommended for this particular dataset.
