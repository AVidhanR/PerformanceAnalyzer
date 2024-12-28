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

# for required data and content for the models
from content_switcher import content_switcher
from pages.footer import footer
import pandas as pd
import streamlit as st

warnings.filterwarnings('ignore')

try:
    def display_dataset_info(dataset):
        # col1, col2, col3 = st.columns([.5, 9, .5], vertical_alignment="center")
        # with col2: tbh, no need this shit
        st.subheader("Dataset Information")

        st.write("Dataset head:")
        st.write(dataset.head())
        st.divider()

        st.write("Dataset description:")
        st.write(dataset.describe())
        st.divider()

        st.write("The complete data provided by the dataset:")
        st.write(dataset)
        st.divider()

        # Display dataset info
        buffer = io.StringIO()
        dataset.info(buf=buffer)
        s = buffer.getvalue()
        st.write("### Dataset attributes with it's data type and memory usages:")
        st.html(f'<pre class="dataset-attr">{s}</pre>')
        st.divider()

    def display_model_results(model_name, y_prediction)->list:
        st.subheader(model_name)
        st.write(content_switcher[model_name])

        cm = confusion_matrix(Y_test, y_prediction)

        # Calculate the performance metrics for the model in percentage
        accuracy = round(accuracy_score(Y_test, y_prediction) * 100, 2)
        precision = round(precision_score(Y_test, y_prediction) * 100, 2)
        recall = round(recall_score(Y_test, y_prediction) * 100, 2)
        f1 = round(f1_score(Y_test, y_prediction, average='weighted') * 100, 2)

        st.html("<b><i>Confusion Matrix: </i></b>")
        st.write(
            pd.DataFrame({
                "0": cm[:, 0],
                "1": cm[:, 1]
            })
        )

        # st.metric() API
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{accuracy}%")
        col2.metric("Precision", f"{precision}%")
        col3.metric("Recall", f"{recall}%")
        col4.metric("F1-Score", f"{f1}%")
        st.divider()
        return accuracy, precision, recall, f1

    title_of_the_project = "performance analysis of different classification algorithms for heart disease prediction"

    st.title(title_of_the_project.title())
    st.sidebar.subheader("Settings")

    # upload the dataset
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        uploaded_dataset = pd.read_csv(uploaded_file)
        dataset = uploaded_dataset.dropna().reset_index(drop=True)
        if ("gender") in uploaded_dataset.columns:
            dataset['gender'] = dataset['gender'].replace({'male': 1, 'female': 0})
        elif "sex" in uploaded_dataset.columns:
            dataset['sex'] = dataset['sex'].replace({'male': 1, 'female': 0})
        st.sidebar.success("Dataset uploaded successfully.")
    else:
        st.error("Please upload a CSV file.")
        footer()
        st.stop()

    # display basic information about the dataset
    display_dataset_info(dataset)

    target_input = st.text_input(label="Enter the valid target variable",placeholder="Enter the valid target variable name", max_chars=20, label_visibility="visible")
    dataset_target_var = target_input.lower()

    if dataset_target_var not in dataset.columns:
        st.error("Please enter a valid target variable name that is available in the dataset.")
        st.stop()

    st.divider()

    # Target distribution
    st.subheader("Target Distribution")
    st.html(f'''
        <details>
            <summary>About</summary>
            The target variable {dataset_target_var} is the presence or absence of heart disease. The value 1 represents the presence of heart disease and the value 0 represents the absence of heart disease.
        </details>
    ''')
    fig, ax = plt.subplots()
    sns.countplot(x=dataset[dataset_target_var], data=dataset, ax=ax)
    st.pyplot(fig)

    st.divider()

    # Split dataset
    predictors = dataset.drop(dataset_target_var, axis=1)
    target = dataset[dataset_target_var]
    X_train, X_test, Y_train, Y_test = train_test_split(predictors, target, test_size=0.25, random_state=0)

    # Model Building and Evaluation 
    st.write("## Model Building and Evaluation")
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, Y_train)
    Y_prediction_nb = nb.predict(X_test)
    accuracy_nb, precision_nb, recall_nb, f1_nb = display_model_results("Naive Bayes", Y_prediction_nb)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, Y_train)
    Y_prediction_knn = knn.predict(X_test)
    accuracy_knn, precision_knn, recall_knn, f1_knn = display_model_results("K-Nearest Neighbors", Y_prediction_knn)

    # Support Vector Machines - Linear
    svm_model = svm.LinearSVC()
    svm_model.fit(X_train, Y_train)
    Y_prediction_svm = svm_model.predict(X_test)
    accuracy_svm, precision_svm, recall_svm, f1_svm = display_model_results("Support Vector Machines", Y_prediction_svm)

    # Neural Networks - MLP
    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    mlp.fit(X_train, Y_train)
    Y_pred_mlp = mlp.predict(X_test)
    accuracy_mlp, precision_mlp, recall_mlp, f1_mlp = display_model_results("Multilayer Perceptron", Y_pred_mlp)

    # Gradient Boosting - the best for now
    gb = GradientBoostingClassifier()
    gb.fit(X_train, Y_train)
    Y_pred_gb = gb.predict(X_test)
    accuracy_gb, precision_gb, recall_gb, f1_gb = display_model_results("Gradient Boosting", Y_pred_gb)

    # Random Forest - n_estimators = 600 and random_state = 42 - default_since
    rf = RandomForestClassifier(n_estimators=700, random_state=42)
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)
    accuracy_rf, precision_rf, recall_rf, f1_rf = display_model_results("Random Forest", Y_pred_rf)

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(X_train, Y_train)
    Y_pred_dt = dt.predict(X_test)
    accuracy_dt, precision_dt, recall_dt, f1_dt = display_model_results("Decision Tree", Y_pred_dt)

    # Logistic Regression
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    Y_pred_lr = lr.predict(X_test)
    accuracy_lr, precision_lr, recall_lr, f1_lr = display_model_results("Logistic Regression", Y_pred_lr)

    # Bar plots for categorical features
    st.subheader(f"Categorical Features vs {dataset_target_var}")
    st.html(f'''
        <details>
            <summary>About</summary>
            Below are the bar plots showing the relationship between categorical features and the target variable '{dataset_target_var}'. These plots help in understanding how different categorical features are distributed with respect to the presence or absence of heart disease.
        </details>
    ''')

    # categorical_features = dataset.columns[:-1].tolist()
    # for feature in categorical_features:
    #     fig, ax = plt.subplots()
    #     sns.barplot(x=feature, y=dataset[dataset_target_var], data=dataset, ax=ax)
    #     st.pyplot(fig)

    st.write('''
        ## Model Comparison
        Below are the comparison of all the models based on the metrics such as Accuracy, Precision, Recall and F1-Score.
    ''')


    algorithms = ["Naive Bayes", "K-Nearest Neighbors", "Support Vector Machines", "Multilayer Perceptron", "Gradient Boosting", "Random Forest", "Decision Tree", "Logistic Regression"]

    accuracy_scores = [accuracy_nb, accuracy_knn, accuracy_svm, accuracy_mlp, accuracy_gb, accuracy_rf, accuracy_dt, accuracy_lr]

    precision_scores = [precision_nb, precision_knn, precision_svm, precision_mlp, precision_gb, precision_rf, precision_dt, precision_lr]

    recall_scores = [recall_nb, recall_knn, recall_svm, recall_mlp, recall_gb, recall_rf, recall_dt, recall_lr]

    f1_scores = [f1_nb, f1_knn, f1_svm, f1_mlp, f1_gb, f1_rf, f1_dt, f1_lr]

    # Plot Accuracy
    st.subheader("Accuracy Score Comparison")
    accuracy_df = pd.DataFrame({
        "Algorithms": algorithms,
        "Accuracy Score": accuracy_scores
    })
    st.bar_chart(accuracy_df.set_index("Algorithms"))
    st.divider()

    # Plot Precision
    st.subheader("Precision Score Comparison")
    precision_df = pd.DataFrame({
        "Algorithms": algorithms,
        "Precision Score": precision_scores
    })
    st.bar_chart(precision_df.set_index("Algorithms"))
    st.divider()

    # Plot Recall
    st.subheader("Recall Score Comparison")
    recall_df = pd.DataFrame({
        "Algorithms": algorithms,
        "Recall Score": recall_scores
    })
    st.bar_chart(recall_df.set_index("Algorithms"))
    st.divider()

    # Plot F1-Score
    st.subheader("F1-Score Comparison")
    f1_df = pd.DataFrame({
        "Algorithms": algorithms,
        "F1-Score": f1_scores
    })
    st.bar_chart(f1_df.set_index("Algorithms"))
    st.divider()

    # Display DataFrame with results
    data = {
        "Algorithm": algorithms,
        "Accuracy": accuracy_scores,
        "Precision": precision_scores,
        "Recall": recall_scores,
        "F1": f1_scores
    }

    df = pd.DataFrame(data)
    df.set_index("Algorithm", inplace=True)
    st.subheader("Comparison of all Performance Metrics")
    st.write(df)
    st.divider()

    st.subheader("Comparison of all Performance Metrics in a single chart")
    st.bar_chart(df)

    def display_best_model(algorithms, accuracy, precision, recall, f1):

        metrics_df = pd.DataFrame({
            "Algorithm": algorithms,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })
        metrics_df["Average"] = metrics_df[["Accuracy", "Precision", "Recall", "F1-Score"]].mean(axis=1)
        best_model = metrics_df.loc[metrics_df["Average"].idxmax()]

        st.subheader("Best Performing Model for Heart Disease Prediction using this dataset")
        st.write(f"**Algorithm:** {best_model['Algorithm']}")
        st.write(f"**Average Score:** {best_model['Average']:.2f}")
        st.write("**Metrics:**")
        st.write(best_model[["Accuracy", "Precision", "Recall", "F1-Score"]])

    display_best_model(algorithms, accuracy_scores, precision_scores, recall_scores, f1_scores)

    footer()
except Exception as e:
    st.error(f"An error occurred: {e}")
    footer()
    st.stop()