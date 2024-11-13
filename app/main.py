# # import the basic packages
# import warnings
# import io
#
# # visualization packages
# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
#
# # classification packages from scikit-learn ~ sklearn
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
# from sklearn.naive_bayes import GaussianNB
# from sklearn import svm
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.linear_model import LogisticRegression
#
# # ignore warnings
# warnings.filterwarnings('ignore')
#
# # project setup
# st.set_page_config(
#     page_title="Performance Analyzer",
#     page_icon=":bar_chart:",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )
#
# # Changed the font style to Open Sans
# st.markdown(
#     """
#     <style>
#     /* Import Open Sans font */
#     @import url('https://fonts.googleapis.com/css2?family=Open+Sans:ital@0;1&display=swap');
#
#      body {
#         font-family: 'Open Sans';
#      }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )
#
# # title of the project
# title_of_the_project = "performance analysis for heart disease prediction"
#
# # set the title of the web app
# st.title(title_of_the_project.title())
# st.sidebar.title(title_of_the_project.title())
# st.sidebar.subheader("Settings")
#
# # upload the dataset
# uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
# if uploaded_file is not None:
#     dataset = pd.read_csv(uploaded_file)
# else:
#     st.error("Please upload a CSV file.")
#     st.stop()
#
# # display basic information about the dataset
# col1, col2, col3 = st.columns([1, 7, 1], vertical_alignment="center")
# with col2:
#     st.subheader("Dataset Information")
#
#     st.write("Dataset head:")
#     st.write(dataset.head())
#
#     st.write("Dataset description:")
#     st.write(dataset.describe())
#
#     st.write("The complete data provided by the dataset:")
#     st.write(dataset)
#
#     # Display dataset info
#     buffer = io.StringIO()
#     dataset.info(buf=buffer)
#     s = buffer.getvalue()
#
#     st.write("Dataset attributes with it's data type and memory usages:")
#     st.text(s)
#
#
# # Target distribution
# # st.subheader("Target Distribution")
# # fig, ax = plt.subplots()
# # sns.countplot(x=dataset["target"], data=dataset, ax=ax)
# # st.pyplot(fig)
#
# # # Bar plots for categorical features
# # st.subheader("Categorical Features vs Target")
# # categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
# # for feature in categorical_features:
# #     fig, ax = plt.subplots()
# #     sns.barplot(x=feature, y=dataset["target"], data=dataset, ax=ax)
# #     st.pyplot(fig)
#
# # Split dataset
# predictors = dataset.drop("target", axis=1)
# target = dataset["target"]
# X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.20, random_state=0)
#
# st.markdown("""<hr style="margin: 0rem;" />""", unsafe_allow_html=True)
#
# # Define function to display results
# def display_results(model_name, y_prediction):
#     st.header(model_name)
#     cm = confusion_matrix(Y_test, y_prediction)
#     accuracy = round(accuracy_score(Y_test, y_prediction) * 100, 2)
#     precision = round(precision_score(Y_test, y_prediction) * 100, 2)
#     recall = round(recall_score(Y_test, y_prediction) * 100, 2)
#
#     # --- Display the Confusion Matrix ---
#     st.subheader("Confusion Matrix")
#     st.write(cm)
#
#     st.subheader("The acquired results,")
#     st.write(f"The accuracy score achieved using {model_name} is: {accuracy} %")
#     st.write(f"The precision score achieved using {model_name} is: {precision} %")
#     st.write(f"The recall score achieved using {model_name} is: {recall} %")
#     st.write(" ")
#     st.markdown("""<hr style="margin: 0rem;" />""", unsafe_allow_html=True)
#
#     return accuracy, precision, recall
#
#
#
# # Now let's build the models and display the results for each model using the # function defined above.
#
#
# # Naive Bayes Classifier (NB) - Gaussian Naive Bayes Classifier (GaussianNB)
# nb = GaussianNB()
# nb.fit(X_train, Y_train)
# Y_prediction_nb = nb.predict(X_test)
# accuracy_nb, precision_nb, recall_nb = display_results("Naive Bayes", Y_prediction_nb)
#
# # K-Nearest Neighbors (KNN) Classifier with k=7 (default)
# knn = KNeighborsClassifier(n_neighbors=7)
# knn.fit(X_train, Y_train)
# Y_prediction_knn = knn.predict(X_test)
# accuracy_knn, precision_knn, recall_knn = display_results("K-Nearest Neighbors", Y_prediction_knn)
#
# # Support Vector Machine (SVM) Classifier with linear kernel (default)
# svm_model = svm.SVC(kernel='linear')
# svm_model.fit(X_train, Y_train)
# Y_prediction_svm = svm_model.predict(X_test)
# accuracy_svm, precision_svm, recall_svm = display_results("Support Vector Machine", Y_prediction_svm)

import pandas as pd
import streamlit as st

# Download latest version
df = pd.read_csv("./dataset/output.csv")
st.title("CVD-DataSet")

# Display the dataframe
st.write("Here is the dataset:")
st.dataframe(df)

# Optionally, display some statistics about the dataset
st.write("Statistics:")
st.write(df.describe())
