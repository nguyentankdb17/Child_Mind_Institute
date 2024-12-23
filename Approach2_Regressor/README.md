# Approach 2: Regressor the `PCIAT-PCIAT_Total` feature

This approach give us the private score of 0.423. You can validate at this [Link](https://www.kaggle.com/code/nguyensytan/cmi123?scriptVersionId=213905366).  

*Note that the notebook we submitted is different from the one in this GitHub repository. In this repository, we aim to specifically demonstrate our approach step by step. However, all the metrics and outputs in both notebooks are identical.*

This document serves to explain our approach. For detailed information about the code and outputs, please refer to the [notebook](approach2_regressor.ipynb) located in the same folder.  
To execute this notebook, simply click the Open in Kaggle button at the top-left corner of the notebook.

## 1. The reason for choosing this approach
The goal of this competition is to predict from this data a participant's Severity Impairment Index (`sii`), a standard measure of problematic internet use.  According to the problem description, this is a multiclass categorical classification task with outputs ranging from 0 to 3: 0 for None, 1 for Mild, 2 for Moderate, and 3 for Severe. Given this structure, our second approach was to use a regressor method to predict the test data, specifically implementing a XGBoost.

## 2. The process of version upgrade
The table below summarizes some of the key submissions that represent significant changes in our work.

| Version     | Submit date | Description | Private score | Public score |
| --------------- | -------- | --------- | --------- | ---------- |
| Version 1 | 12/112/2024 | First submission (init the simple model with stratifiedKFold) | 0.401 | 0.418 |
| Version 5 |  |  |  |  |
| Version 18 |  |  |  |  |
| Version 21 |  |  |  |  |
| Version 27 |  |  |  |  |
| Version 29 |  |  |  |  |

## 3. Methodology
### 3.1. Data Selection & Data Cleaning
- As in the previous approach, we remove the samples which miss output label to avoid noise points and more efficient computation.
- Along with that we also drop the features which not appear in test set, only keep PCIAT_PCIAT-Total to predict
### 3.2. Feature Engineering
- Firstly, we drop features that only appear in train set and only keep the PCIAT_PCIATTotal feature to predict
- After that we Convert from PCIAT-PCIAT_Total to sii and define a mapping function to convert the corresponding sii
- Finally, we make encode the categorical features
#### 3.2.1 Feature Extraction
- In this step, we check the correlation to PCIAT_PCIAT-Total (|correlation| > 0.05)
- So on we just keep the most importance features (correlation >0.05 or < -0.05) and drop features with more than 50% missing values
### 3.3. Model Selection
- We visualize the output dataframe to examine its distribution. The visualization clearly reveals that the output distribution is imbalanced, with significant differences in the frequency of category labels.
  
  ![Visualization Result](https://i.imgur.com/SaIFaMd.png)

-  To address this issue, we selected the XGBoost from sklearn to train the model
-  XGBoost Extreme Gradient Boosting) is a powerful and efficient machine learning algorithm designed for structured data. It is an implementation of gradient-boosted decision trees, optimized for speed and performance
-  XGBoost operates by sequentially building decision trees, where each new tree attempts to correct the errors of the previous ones. It combines these trees to make predictions, improving accuracy over time. The algorithm uses gradient descent to minimize the loss function and includes regularization techniques to prevent overfitting
-  One key advantage of XGBoost is its ability to handle missing values automatically by learning the best direction to take in a decision tree when encountering them. This 
 liminates the need for explicit imputation, making it a robust choice for datasets with incomplete information. Additionally, XGBoost supports parallel processing, which significantly speeds up training on large
datasets
### 3.4. Model Optimization & Training
- The eval metric is quadratic weighted kappa as mentioned in competition description
- After implementing the model, we decide to chose is Bayesian Optimization is our optimization method - which is wellsuited for finding the best hyperparameters within a defined range
-  After run over 100 iterations we got the best params corresponding to the best score. Then I apply it to the model and get the evaluation score at 0.467 very high score so I use this model to make the final predictions
## 4. Model Performance
We evaluate our model using cross_val_score from sklearn with cv=skf
- Quadratic weighted kappa score
```
QWK Scores: [0.42507361 0.52498236 0.44261374 0.50300074 0.43963159]
----> Mean QWK Score: 0.46706040837853724
```
- Some additional metrics: accuracy score, precision score, f1 score, recall score
```
Mean Accuracy Score: ?
Mean Precision Score: ?
Mean F1 Score: ?
Mean Recall Score: ?
```
## 5. Result
![Best Result](https://imgur.com/w5RDHOB.png)

