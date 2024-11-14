# type: ignore
import warnings
import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# classification packages from scikit-learn ~ sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings('ignore')

def display_results(model_name, y_prediction):
    st.subheader(model_name)
    cm = confusion_matrix(Y_test, y_prediction)
    accuracy = round(accuracy_score(Y_test, y_prediction) * 100, 2)
    precision = round(precision_score(Y_test, y_prediction) * 100, 2)
    recall = round(recall_score(Y_test, y_prediction) * 100, 2)
    f1 = round(f1_score(Y_test, y_prediction, average='weighted') * 100, 2)

    st.write("Confusion Matrix: ")
    st.write(cm)

    st.write(f"""
    The acquired results are,
    - The accuracy score achieved using {model_name} is: **{accuracy}%**
    - The precision score achieved using {model_name} is: **{precision}%**
    - The recall score achieved using {model_name} is: **{recall}%**
    - The F1-Score achieved using {model_name} is: **{f1}%**
    ----
    """)
    return accuracy, precision, recall

st.markdown(
    """
    <style>
     body, p, h1, h2, h3, h4, h5, li, ul, ol{
        font-family: 'Cascadia Code';
     }
    </style>
    """,
    unsafe_allow_html=True,
)

title_of_the_project = "performance analysis for heart disease prediction"

st.title(title_of_the_project.title())
st.sidebar.title(title_of_the_project.title())
st.sidebar.subheader("Settings")

# upload the dataset
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
else:
    st.error("Please upload a CSV file.")
    st.stop()

# display basic information about the dataset
col1, col2, col3 = st.columns([.5, 7.5, .5], vertical_alignment="center")
with col2:
    st.subheader("Dataset Information")

    st.write("Dataset head:")
    st.write(dataset.head())

    st.write("Dataset description:")
    st.write(dataset.describe())

    st.write("The complete data provided by the dataset:")
    st.write(dataset)

    # Display dataset info
    buffer = io.StringIO()
    dataset.info(buf=buffer)
    s = buffer.getvalue()

    st.write("Dataset attributes with it's data type and memory usages:")
    st.text(s)

    # Target distribution
    st.subheader("Target Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x=dataset["cardio"], data=dataset, ax=ax)
    st.pyplot(fig)

# # # Bar plots for categorical features - need to figure out here.
# # st.subheader("Categorical Features vs cardio")
# # categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
# # for feature in categorical_features:
# #     fig, ax = plt.subplots()
# #     sns.barplot(x=feature, y=dataset["cardio"], data=dataset, ax=ax)
# #     st.pyplot(fig)

# Split dataset
predictors = dataset.drop("cardio", axis=1)
target = dataset["cardio"]
X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)

nb = GaussianNB()
nb.fit(X_train, Y_train)
Y_prediction_nb = nb.predict(X_test)
accuracy_nb, precision_nb, recall_nb = display_results("Naive Bayes", Y_prediction_nb)

# K-Nearest Neighbors (KNN) Classifier with k=7 (default)
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, Y_train)
Y_prediction_knn = knn.predict(X_test)
accuracy_knn, precision_knn, recall_knn = display_results("K-Nearest Neighbors", Y_prediction_knn)

# Support Vector Machines - Linear
svm_model = svm.LinearSVC()
svm_model.fit(X_train, Y_train)
Y_prediction_svm = svm_model.predict(X_test)
accuracy_svm, precision_svm, recall_svm = display_results("Support Vector Machine", Y_prediction_svm)

# Neural Networks - MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
mlp.fit(X_train, Y_train)
Y_pred_mlp = mlp.predict(X_test)
accuracy_mlp, precision_mlp, recall_mlp = display_results("Multilayer Perceptron (Neural Networks)", Y_pred_mlp)

# Gradient Boosting - the best for now
gb = GradientBoostingClassifier()
gb.fit(X_train, Y_train)
Y_pred_gb = gb.predict(X_test)
accuracy_gb, precision_gb, recall_gb = display_results("Gradient Boosting", Y_pred_gb)

# Random Forest - n_estimators = 600 and random_state = 42 - default_since
rf = RandomForestClassifier(n_estimators=400, random_state=32)
rf.fit(X_train, Y_train)
Y_pred_rf = rf.predict(X_test)
accuracy_rf, precision_rf, recall_rf = display_results("Random Forest", Y_pred_rf)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, Y_train)
Y_pred_dt = dt.predict(X_test)
accuracy_dt, precision_dt, recall_dt = display_results("Decision Tree", Y_pred_dt)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, Y_train)
Y_pred_lr = lr.predict(X_test)
accuracy_lr, precision_lr, recall_lr = display_results("Logistic Regression", Y_pred_lr)