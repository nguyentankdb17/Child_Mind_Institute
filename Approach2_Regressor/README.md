# Approach 2: Regressor the `PCIAT-PCIAT_Total` feature

This approach give us the private score of 0.423 and highest public score of 0.460. You can validate at this [Link](https://www.kaggle.com/code/nguyensytan/cmi123?scriptVersionId=213905366).  

*Note that the notebook we submitted is different from the one in this GitHub repository. In this repository, we aim to specifically demonstrate our approach step by step. However, all the metrics and outputs in both notebooks are identical.*

This document serves to explain our approach. For detailed information about the code and outputs, please refer to the [notebook](approach2_regressor.ipynb) located in the same folder.  
To execute this notebook, simply click the `Open in Kaggle` button at the top-left corner of the notebook.

## 1. The reason for choosing this approach
We find out that the ouput `sii` is derived from mapping the `PCIAT_PCIAT-Total` feature - feature that we have dropped in the first approach. The `PCIAT_PCIAT-Total` is a numerical feature which ranging from 0-100 so this lead us to approach 2: use a regressor method to predict the `PCIAT_PCIAT-Total` feature and map them back to `sii`, specifically implementing a XGBoost Regressor model.

## 2. The process of version upgrade
The table below summarizes some of the key submissions that represent significant changes in our work.

| Version     | Submit date | Description | Private score | Public score |
| --------------- | -------- | --------- | --------- | ---------- |
| Version 1 | 14/12/2024 | First regressor approach submission (init with xgboost regressor) | 0.400 | 0.444 |
| Version 6 | 15/12/2024 | 1st selected - (increase correlation threshold to 0.11) | 0.422  | 0.458  |
| Version 12 | 16/12/2024  | Try out LightGBM Regressor  | 0.438 | 0.437  |
| Version 14 | 16/12/2024  | Set correlation threshold to 0.05  | 0.429  | 0.450  |
| Version 21 | 18/12/2024  | Change hyperparams tuning strategy from RandomSearchCV to Bayes Optim  | 0.424  |  0.459 |
| Version 24 | 19/12/2024  | 2nd selected - version with highest public score  | 0.423  | 0.460  |

## 3. Methodology
### 3.1. Data Selection & Data Cleaning
- As in the previous approach, we remove the samples which miss output label to avoid noise points and more efficient computation.
- Along with that we also drop the features which not appear in test set, only keep `PCIAT_PCIAT-Total` to predict
### 3.2 Define mapping function
We define a mapping function to convert the `PCIAT_PCIAT-Total` value into the corresponding `sii` category as follows:
- Values less than 30 are mapped to `0` in `sii`.
- Values between 31 and 49 are mapped to `1` in `sii`.
- Values between 50 and 79 are mapped to `2` in `sii`.
- Values 80 and above are mapped to `3` in `sii`.
### 3.3. Feature Engineering
#### 3.3.1 Feature Extraction
- In this step, we check the correlation to PCIAT_PCIAT-Total (|correlation| > 0.05)
- So on we just keep the most importance features (correlation >0.05 or < -0.05) and drop features with more than 50% missing values
#### 3.3.2 Feature Encoding
Different from previous approach, in this approach we chose to use Label Encoder to handle categorical features, we handle them by filling the missing cells with `0` and mapping the season names to numerical values: `Spring` as `1`, `Summer` as `2`, `Fall` as `3`, and `Winter` as `4`.
### 3.4. Model Selection
- We select the XGBoost Regressor to train the model. XGBoost (Extreme Gradient Boosting) is a powerful and efficient machine learning algorithm designed for structured data. It is an implementation of gradient-boosted decision trees, optimized for speed and performance.
- XGBoost operates by sequentially building decision trees, where each new tree attempts to correct the errors of the previous ones. It combines these trees to make predictions, improving accuracy over time. The algorithm uses gradient descent to minimize the loss function and includes regularization techniques to prevent overfitting.
- One key advantage of XGBoost is its ability to handle missing values automatically by learning the best direction to take in a decision tree when encountering them. This eliminates the need for explicit imputation, making it a robust choice for datasets with incomplete information. Additionally, XGBoost supports parallel processing, which significantly speeds up training on large datasets.
### 3.5. Model Optimization & Training
- For the splitting strategy, we use `StratifiedKFold` from `sklearn` with `n_splits=5`, which is more efficient than train_test_split  and the evaluation metric is still qwk because we have converted the PCIAT-PCIAT_Total to the sii format
- After implementing the model, we decide to chose is Bayesian Optimization is our optimization method - which is wellsuited for finding the best hyperparameters within a defined range
-  After run over 100 iterations we got the best params corresponding to the best score. Then we apply it to the model and train the model.
## 4. Model Performance
We evaluate our model using cross_val_score from sklearn with the split strategy declared above
- Quadratic weighted kappa score
```
QWK Scores: [0.42507361 0.52498236 0.44261374 0.50300074 0.43963159]
----> Mean QWK Score: 0.46706040837853724
```
The evaluation score is good so we chose to use this model to make the final predictions
## 5. Result
![Best Result](https://imgur.com/w5RDHOB.png)
The public score met our expectation so that we decided to choose this approach to to be evaluated for the final leaderboard.
