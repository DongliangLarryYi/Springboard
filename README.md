# Springboard
 
This is using Random Forest and Logistic Regression to forecast repayment of loan applicants

**home_credit.py**
This python code read the training data and explored the data.
We trained logistic regression and random forests model.
The Random forest model is saved to "classifier.pkl" and used on the application



```
For alpha 0.001, cross validation AUC score 0.7534614416480928
For alpha 0.01, cross validation AUC score 0.7411513592164468
For alpha 0.1, cross validation AUC score 0.7063281322965389
For alpha 1.0, cross validation AUC score 0.5
For alpha 10.0, cross validation AUC score 0.5
For alpha 100.0, cross validation AUC score 0.5
For alpha 1000.0, cross validation AUC score 0.5
For alpha 10000.0, cross validation AUC score 0.5
```


**prediction.py**
This python code is ran on streamlit cloud. The model is based on random forest
It generates a web application which is used to predict the repayment of an applicant

