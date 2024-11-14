## Performance Metrics

Below are the list of **Performance Metrics** used in this project,

### `accuracy`

Accuracy is a performance metric used to evaluate the effectiveness of a machine learning model. It measures the proportion of correct predictions made by the model compared to the total number of predictions. Essentially, it tells us how often the model is right.
- General Formula for Accuracy
    - The formula to calculate accuracy is:
$$\text{Accuracy}=\frac{\text{Total Number of Predictions}}{\text{Number of Correct Predictions}}×\text{100%}$$
    - This formula gives the accuracy as a percentage, indicating how many predictions were correct out of all predictions made.
- The Accuracy of a classification model can be expressed as:
$$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}}×\text{100%}$$

- Real-Life Example
  - Imagine a hospital that uses a machine learning model to predict whether patients have a certain disease based on test results.
    - Total Patients Tested: 100
    - Correct Predictions: 90 (80 true positives and 10 true negatives)
    - Incorrect Predictions: 10 (5 false positives and 5 false negatives)
  - Using the accuracy formula:
    $$\text{Accuracy}=\frac{\text{90}}{\text{100}}×\text{100%}$$
    $$\text{Accuracy}=\text{90%}$$
    - In this case, the model has an accuracy of 90%, indicating it correctly identified the disease in most patients tested. However, it's important to note that high accuracy can be misleading if the dataset is imbalanced (e.g., if most patients do not have the disease).

### `precision`

Precision is a performance metric used to evaluate the accuracy of positive predictions made by a machine learning model. It measures how many of the predicted positive cases were actually correct. In other words, precision tells us how reliable the positive predictions are.
- Formula for Precision
  - The formula to calculate precision is:
$$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives}+\text{False Positives}}×\text{100%}$$
  - This formula gives precision as a percentage, indicating the proportion of correct positive predictions out of all positive predictions made.

- Real-Life Example
  - Imagine a medical test used to diagnose a disease:
    - Total Positive Predictions: 50 (the test says these patients have the disease)
    - True Positives: 45 (patients who actually have the disease)
    - False Positives: 5 (patients who do not have the disease but were incorrectly identified by the test)
  - Using the precision formula:
    $$\text{Precision}=\frac{\text{45}}{\text{45} + \text{5}}×{\text{100%}}$$
    $$\text{Precision}=\text{90%}$$
    - In this case, the precision of the medical test is 90%. This indicates that when the test predicts a patient has the disease, there is a high likelihood that they actually do.
    - High precision is particularly important in medical diagnoses to minimize unnecessary anxiety and treatment for patients who do not have the disease.
### `sensitivity or recall`


### `f1 score`

