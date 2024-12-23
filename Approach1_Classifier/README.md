# Approach 1: Classify the ouput `sii` feature
The table below summarizes some of the key submissions that represent breakthroughs in our work.

| Version     | Submit date | Description | Private score | Public score |
| --------------- | -------- | --------- | --------- | ---------- |
| Version 1 | 01/12/2024 | Init the simple model with random forest classifier (first submission) | 0.296 | 0.268 |
| Version 5 | 01/12/2024 | Using hyperparams tuning to find out the best params | 0.390 | 0.413 |
| Version 18 | 07/12/2024 | Try out TabNet Classifier (not efficient) | 0.276 | 0.300 |
| Version 21  |  10/12/2024 | Handle multiclass catgorical int features | 0.458 | 0.412 |
| Version 29 | 12/12/2024 | Highest private score (only tuning hyperparams from version 22) | 0.471 | 0.412 |

Our first approach to this problem was to classify the output `sii` feature. We did it by following steps
## 1. Data Selection & Data Cleaning
- In this approach, we didn't use the actigraphy files due to their complexity and high missing data ratio (approximately 75%). Instead, we focused on the train and test CSV files, using the pandas library to read them as dataframes.
- We drop the features without output label to avoid noise points and more efficient computation
## 2. Feature Engineering
### 2.1 Feature Extraction
- We dropped features that only appeared in the training set, which were the 'PCIAT' features. For the remaining features, we used matplotlib to visualize their missing value ratios
- We discovered that many features had high missing ratios. To reduce unefficient features, we dropped any features that were missing more than 50% of their values.
### 2.2 Feature Categorization
- After discover the data dictionary file, we find out that there’re 4 types of features
    - Categorical features in object type: feature has ‘Season’ in name
    - Multiclass categorical features in number type: `BIA-BIA_Activity_Level_num`, `FGC-FGC_GSD_Zone`,`FGC-FGC_GSND_Zone`,`BIA-BIA_Frame_num`, `PreInt_EduHx-computerinternet_hoursday`
    - Binary features: also categorical features but only have value 0 or 1
    - Numerical features: remaining features
### 2.3 Imputation
- After categorize the features, we use imputation to each type of features:
    - For the categorical features and binary features: we filll out the NaN value by the mode value (most frequently occurring value)
    - For numerical features: we fill out the NaN value by the mean value
### 2.4 Feature Scaling
- Then, we apply feature scaling to numerical features using Standard Scaler from sklearn, a preprocessing technique that standardizes features by removing the mean and scaling them to unit variance, ensuring that each feature has a mean of 0 and a standard deviation of 1.
### 2.5 Feature Encoding
- After that, we apply one hot encoding to encode the categorical features, after one hot there are a few features which only appear in train set, so we continue drop them to avoid noise points
## 3. Model Selection
- We visualize the output dataframe to examine its distribution. The visualization clearly reveals that the output distribution is imbalanced, with significant differences in the frequency of category labels.
  
  ![Visualization Result](https://i.imgur.com/f2dQG0Q.png)

-  To address this issue, we selected the Random Forest Classifier from sklearn to train the model, as it is particularly effective for handling imbalanced data.
-  Random Forest is an ensemble learning method based on decision trees. It builds multiple decision trees during training and combines their predictions to produce a more robust and accurate result. For classification tasks, Random Forest aggregates the predictions of individual trees through majority voting, ensuring stability and reducing overfitting.
- When dealing with imbalanced data, Random Forest can address the issue by assigning higher weights to minority classes, ensuring they have a stronger influence during tree construction.
- By leveraging these characteristics, Random Forest Classifier provides a robust solution for imbalanced datasets, making it an ideal choice for our problem.
## 4. Model Optimization & Training
- The eval metric is quadratic weight kappa as mentioned in competition description.
- We decide to optimize the model by using GridSearch CV - a model optimization method from sklearn to find out the best hyperparameters.  The hyperparameters sets in gridsearch cv were selected based on insights gained from our previous submissions.
- After fitting all possible tuples the grid search gives us the best params corresponding to the best score. Now we will assign these hyperparameters back to the model and training it:
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
## 5. Model Evaluation
- We evaluate our model by quadratic weight kappa score, f1 score, recall score, 

