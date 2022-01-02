# Springboard
 
This is using Random Forest and Logistic Regression to forecast repayment of loan applicants

**home_credit.py**
This python code read the training data and explored the data.
We trained logistic regression and random forests model.
The Random forest model is saved to "classifier.pkl" and used on the application

Exploratory Data Analysis

![This is an image](https://github.com/DongliangLarryYi/Springboard/blob/master/Dependent%20variable%20distribution.png)

Most borrowed repayed their loans ('1')

![This is an image](https://github.com/DongliangLarryYi/Springboard/blob/master/Employment%20outlier.png)

Some applicants have wrong employment data


Logistic regression
Cross validation
```
For alpha 0.001, cross validation AUC score 0.7534614416480928
For alpha 0.01, cross validation AUC score 0.7411513592164468
For alpha 0.1, cross validation AUC score 0.7063281322965389
For alpha 1.0, cross validation AUC score 0.5
For alpha 10.0, cross validation AUC score 0.5
For alpha 100.0, cross validation AUC score 0.5
For alpha 1000.0, cross validation AUC score 0.5
For alpha 10000.0, cross validation AUC score 0.5```
```
The Optimal C value is: 0.0001
For best alpha 0.0001, The Train AUC score is 0.7634454663526145
For best alpha 0.0001, The Cross validated AUC score is 0.756648478021221
For best alpha 0.0001, The Test AUC score is 0.7588733892503452
The test AUC score is : 0.7588733892503452
The percentage of misclassified points 28.55% ```


Random Forests
Cross validation
```
For n_estimators 200, max_depth 5 cross validation AUC score 0.7350981362833978
For n_estimators 200, max_depth 7 cross validation AUC score 0.7450089599835531
For n_estimators 500, max_depth 5 cross validation AUC score 0.7354467426293456
For n_estimators 500, max_depth 7 cross validation AUC score 0.7448232008288938
For n_estimators 1000, max_depth 5 cross validation AUC score 0.7353487556726741
For n_estimators 1000, max_depth 7 cross validation AUC score 0.744935499846709
```
```
The optimal values are: n_estimators 200, max_depth 7 
For best n_estimators 200 best max_depth 7, The Train AUC score is 0.7732236707662539
For best n_estimators 200 best max_depth 7, The Validation AUC score is 0.7450089599835531
For best n_estimators 200 best max_depth 7, The Test AUC score is 0.7438669588286078
The test AUC score is : 0.7438669588286078
The percentage of misclassified points 08.07% ```

**prediction.py**
This python code is ran on streamlit cloud. The model is based on random forest
It generates a web application which is used to predict the repayment of an applicant

