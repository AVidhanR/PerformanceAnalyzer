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