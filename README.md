### Performance Analysis For Heart Disease Prediction

- Get a hold of the packages and libraries that are needed for the project from [here](PACKAGES.md)
- This project compares 7 supervised machine learning algorithms (classification algorithms) for the heart disease dataset and gives the best performed classification algorithm for heart disease prediction.
- The type of heart disease we see here is known as `Cardeovascular disease (CVD)`
- The flow of the project is given below
  > coming soon

### Abstract

- Machine learning technologies have been proven to provide the best solutions to healthcare problems and biomedical communities.It also helps in the early prediction of the disease.
- Symptoms of the disease can be controlled, and proper treatment of the disease can be done due to the early prediction of the disease.
- **The number of deaths due to heart attacks is increasing exponentially.** Thus, **machine learning approaches can be used in the early prediction of heart disease.**
- Different _supervised machine-learning techniques_ like `K-Nearest Neighbors`, `Naive Bayes`, `Support Vector Machines`, `Neural Networks`, `Random Forest Classifier`, `Decision Tree Classifier` and the `Gradient Boosting Classifier` are used for predicting heart disease using a dataset that was collected from the **`University of California, Irvine (UCI) Machine Repository`**
- Among all other supervised classifiers, the results depict that the `Gradient Boosting Classifier` was better in terms of performance metrics like accuracy, precision, and sensitivity.
> [!IMPORTANT] 
> **Keywords:** Heart disease, Machine learning, University of California, Irvine (UCI) Machine Repository, K-Nearest neighbors, Naive Bayes, support vector machines, neural networks, Random Forest Classifier, Decision Tree Classifier, Gradient Boosting Classifier.

---

### To run this project locally

- Make sure to have a `python interpreter` with version **`3.12`**
- After `cloning the repo` by git\/github desktop open the respective `IDE` (VS-Code or PyCharm)

```bash
pip install -r requirements.txt
```

- After entering the above command hit enter, which then installs `streamlit` `seaborn` `pandas` `matplotlib` `scikit-learn` with appropriate versions as mentioned in the `requirements.txt`
- In order to run the web app,

```bash
streamlit run app/main.py
```

- It might automatically open the web app directly on to your default browser or open the `localhost` by typing as `localhost:8501`
- In order to get the analysis part, upload the `University of California, Irvine (UCI) Machine Repository - heart disease` dataset by downloading it.
- Change: using the latest dataset from here https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
