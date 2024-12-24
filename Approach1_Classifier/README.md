# Approach 1: Classify the ouput `sii` feature

This approach give us the highest private score of 0.471. You can validate at this [Link](https://www.kaggle.com/code/nguyensytan/cmi-submit).  

*Note that the notebook we submitted is different from the one in this GitHub repository. In this repository, we aim to specifically demonstrate our approach step by step. However, all the metrics and outputs in both notebooks are identical.*

This document serves to explain our approach. For detailed information about the code and outputs, please refer to the [notebook](approach1_classifier.ipynb) located in the same folder.  
To execute this notebook, simply click the Open in Kaggle button at the top-left corner of the notebook.

## 1. The reason for choosing this approach
The goal of this competition is to predict from this data a participant's Severity Impairment Index (`sii`), a standard measure of problematic internet use.  According to the problem description, this is a multiclass categorical classification task with outputs ranging from 0 to 3: 0 for None, 1 for Mild, 2 for Moderate, and 3 for Severe. Given this structure, our first approach was to use a classifier method to predict the test data, specifically implementing a Random Forest Classifier.

## 2. The process of version upgrade
The table below summarizes some of the key submissions that represent significant changes in our work.

| Version     | Submit date | Description | Private score | Public score |
| --------------- | -------- | --------- | --------- | ---------- |
| Version 1 | 30/11/2024 | First submission (init the simple model with random forest classifier) | 0.296 | 0.268 |
| Version 5 | 02/12/2024 | Using hyperparams tuning to find out the best params | 0.382 | 0.405 |
| Version 18 | 07/12/2024 | Try out TabNet Classifier (not efficient) | 0.276 | 0.300 |
| Version 21  |  10/12/2024 | Handle multiclass catgorical int features | 0.439 | 0.411 |
| Version 27 | 12/12/2024 | Change strategy in filling out NaN values | 0.458 | 0.412 |
| Version 29 | 12/12/2024 | Highest score version (only tuning hyperparams from version 27) | 0.471 | 0.412 |

## 3. Methodology
### 3.1. Data Selection & Data Cleaning
- In this approach, we didn't use the actigraphy files due to their complexity and high missing data ratio (approximately 75%). Instead, we focused on the train and test CSV files, using the pandas library to read them as dataframes.
- We drop the features without output label to avoid noise points and more efficient computation
### 3.2. Feature Engineering
#### 3.2.1 Feature Extraction
- We dropped features that only appeared in the training set, which were the 'PCIAT' features. For the remaining features, we used matplotlib to visualize their missing value ratios
- We discovered that many features had high missing ratios. To reduce unefficient features, we dropped any features that were missing more than 50% of their values.
#### 3.2.2 Feature Categorization
- After discover the data dictionary file, we find out that there’re 4 types of features
    - Categorical features in object type: feature has ‘Season’ in name
    - Multiclass categorical features in number type: `BIA-BIA_Activity_Level_num`, `FGC-FGC_GSD_Zone`,`FGC-FGC_GSND_Zone`,`BIA-BIA_Frame_num`, `PreInt_EduHx-computerinternet_hoursday`
    - Binary features: also categorical features but only have value 0 or 1
    - Numerical features: remaining features
#### 3.2.3 Imputation
- After categorize the features, we use imputation to each type of features:
    - For the categorical features and binary features: we filll out the NaN value by the mode value (most frequently occurring value)
    - For numerical features: we fill out the NaN value by the mean value
#### 3.2.4 Feature Scaling
- Then, we apply feature scaling to numerical features using Standard Scaler from sklearn, a preprocessing technique that standardizes features by removing the mean and scaling them to unit variance, ensuring that each feature has a mean of 0 and a standard deviation of 1.
#### 3.2.5 Feature Encoding
- After that, we apply one hot encoding to encode the categorical features, after one hot there are a few features which only appear in train set, so we continue drop them to avoid noise points
### 3.3. Model Selection
- We visualize the output dataframe to examine its distribution. The visualization clearly reveals that the output distribution is imbalanced, with significant differences in the frequency of category labels.
  
  ![Visualization Result](https://i.imgur.com/f2dQG0Q.png)

-  To address this issue, we selected the Random Forest Classifier from sklearn to train the model, as it is particularly effective for handling imbalanced data.
-  Random Forest is an ensemble learning method based on decision trees. It builds multiple decision trees during training and combines their predictions to produce a more robust and accurate result. For classification tasks, Random Forest aggregates the predictions of individual trees through majority voting, ensuring stability and reducing overfitting.
- When dealing with imbalanced data, Random Forest can address the issue by assigning higher weights to minority classes, ensuring they have a stronger influence during tree construction.
- By leveraging these characteristics, Random Forest Classifier provides a robust solution for imbalanced datasets, making it an ideal choice for our problem.
### 3.4. Model Optimization & Training
- The eval metric is quadratic weighted kappa as mentioned in competition description.
- We decide to optimize the model by using GridSearch CV - a model optimization method from sklearn to find out the best hyperparameters.  The hyperparameters sets in gridsearch cv were selected based on insights gained from our previous submissions.
- After fitting all possible combinations, the grid search provides us with the best parameters corresponding to the highest score. Then, we will assign these hyperparameters back to the model and train it:
```
final_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    max_features='sqrt',
    min_samples_split=5,
    min_samples_leaf=1,
    class_weight='balanced',
    random_state=42,
)

final_model.fit(X, y)
```
## 4. Model Performance
We evaluate our model using cross_val_score from sklearn with cv=3
- Quadratic weighted kappa score
```
QWK Scores: [0.41271546 0.43165367 0.43104488]
----> Mean QWK Score: 0.42513800529318296
```
- Some additional metrics: accuracy score, precision score, f1 score, recall score
```
Mean Accuracy Score: 0.5672514619883041
Mean Precision Score: 0.3652830065038981
Mean F1 Score: 0.3701287213885644
Mean Recall Score: 0.3781062639306915
```
## 5. Result
![Best Result](https://i.imgur.com/hZz2Abi.png)
After submit to competiton we got the public score of 0.412 and  that didn’t meet our expectations. So we moved to approach 2 - regressor the `PCIAT_PCIAT-Total`.

