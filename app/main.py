# import the basic packages

import warnings
import io

# visualization packages
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# classification packages from scikit-learn ~ sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# ignore warnings
warnings.filterwarnings('ignore')

# set the title of the web app
st.title("Heart Disease Classification")
st.sidebar.title("Heart Disease Classification")
st.sidebar.subheader("Settings")

# upload the dataset
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
else:
    st.error("Please upload a CSV file.")
    st.stop()

# display basic information about the dataset
st.subheader("Dataset Information")
st.write(dataset.head())
st.write(dataset.describe())

# Display dataset info
buffer = io.StringIO()
dataset.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Target distribution
st.subheader("Target Distribution")
fig, ax = plt.subplots()
sns.countplot(x="target", data=dataset, ax=ax)
st.pyplot(fig)

# Bar plots for categorical features
st.subheader("Categorical Features vs Target")
categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
for feature in categorical_features:
    fig, ax = plt.subplots()
    sns.barplot(x=feature, y="target", data=dataset, ax=ax)
    st.pyplot(fig)

# Split dataset
predictors = dataset.drop("target", axis=1)
target = dataset["target"]
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

# Define function to display results
def display_results(model_name, y_prediction):
    cm = confusion_matrix(Y_test, y_prediction)
    accuracy = round(accuracy_score(Y_test, y_prediction) * 100, 2)
    precision = round(precision_score(Y_test, y_prediction) * 100, 2)
    recall = round(recall_score(Y_test, y_prediction) * 100, 2)

    st.header(f"{model_name}")
    st.write(f"The accuracy score achieved using {model_name} is: {accuracy} %")
    st.write(f"The precision score achieved using {model_name} is: {precision} %")
    st.write(f"The recall score achieved using {model_name} is: {recall} %")
    st.write(" ")
    return accuracy, precision, recall

"""
Now let's build the models and display the results for each model using the function defined above.
"""

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_prediction_nb = nb.predict(X_test)
accuracy_nb, precision_nb, recall_nb = display_results("Naive Bayes", Y_prediction_nb)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
accuracy_knn, precision_knn, recall_knn = display_results("K-Nearest Neighbors", Y_pred_knn)
