import streamlit as st
import numpy as np
from io import StringIO
import pandas as pd

st.title('Prediction of repayment')

uploaded_file = st.file_uploader("Choose a file)")
if uploaded_file is not None:
     # Can be used wherever a "file-like" object is accepted:
     input_data = pd.read_csv(uploaded_file)
     #st.write(dataframe)

# Save the random forest model
import joblib
loaded_model = joblib.load("classifier.pkl")

# retrieve the list of selected variables
import pickle
with open("selected_features.txt", "rb") as fp:   # Unpickling
    b = pickle.load(fp)

# input csv file
#import pandas as pd
#input_data = pd.read_csv("X_test_final.csv")

# do the prediction based on the loaded model
y_pred = loaded_model.predict(input_data[b])
st.write("""
# Here are forecast:
""")
ID_No = 1
for x in y_pred:
    if x == 0:
        st.write('Borrower #', ID_No, "is expected to repay the loan")
    else:
        st.write('WARNING!!! Borrower #', ID_No, "is expected to default the loan" )
    ID_No = ID_No + 1


