## Performance Metrics

Below are the list of **Performance Metrics** used in this project, (Make sure go through the definitions and examples for better understanding).

### `accuracy`

Accuracy is a performance metric used to evaluate the effectiveness of a machine learning model. It measures the proportion of correct predictions made by the model compared to the total number of predictions. Essentially, it tells us how often the model is right.

- General Formula for Accuracy

  - The formula to calculate accuracy is:
    $$\text{Accuracy}=\frac{\text{Total Number of Predictions}}{\text{Number of Correct Predictions}}×\text{100\%}$$
  - This formula gives the accuracy as a percentage, indicating how many predictions were correct out of all predictions made.
  - The Accuracy of a classification model can be expressed as:
    $$\text{Accuracy} = \frac{\text{True Positives} + \text{True Negatives}}{\text{Total Samples}}×\text{100\%}$$

- Real-Life Example
  - Imagine a hospital that uses a machine learning model to predict whether patients have a certain disease based on test results.
    - Total Patients Tested: 100
    - Correct Predictions: 90 (80 true positives and 10 true negatives)
    - Incorrect Predictions: 10 (5 false positives and 5 false negatives)
  - Using the accuracy formula:
    $$\text{Accuracy}=\frac{\text{90}}{\text{100}}×\text{100\%}$$
    $$\text{Accuracy}=\text{90\%}$$
  - In this case, the model has an accuracy of 90%, indicating it correctly identified the disease in most patients tested. However, it's important to note that high accuracy can be misleading if the dataset is imbalanced (e.g., if most patients do not have the disease).

### `precision`

Precision is a performance metric used to evaluate the accuracy of positive predictions made by a machine learning model. It measures how many of the predicted positive cases were actually correct. In other words, precision tells us how reliable the positive predictions are.

- Formula for Precision

  - The formula to calculate precision is:
    $$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives}+\text{False Positives}}×\text{100\%}$$
  - This formula gives precision as a percentage, indicating the proportion of correct positive predictions out of all positive predictions made.

- Real-Life Example
  - Imagine a medical test used to diagnose a disease:
    - Total Positive Predictions: 50 (the test says these patients have the disease)
    - True Positives: 45 (patients who actually have the disease)
    - False Positives: 5 (patients who do not have the disease but were incorrectly identified by the test)
  - Using the precision formula:
    $$\text{Precision}=\frac{\text{45}}{\text{45} + \text{5}}×{\text{100\%}}$$
    $$\text{Precision}=\text{90\%}$$
  - In this case, the precision of the medical test is 90%. This indicates that when the test predicts a patient has the disease, there is a high likelihood that they actually do.
  - High precision is particularly important in medical diagnoses to minimize unnecessary anxiety and treatment for patients who do not have the disease.

### `sensitivity or recall`

Sensitivity, also known as Recall or the **True Positive Rate**, is a performance metric that measures how well a model identifies positive instances. \
It answers the question: "Of all the actual positive cases, how many did the model correctly identify?" A high sensitivity means the model is good at detecting positive cases.

- Formula for Sensitivity (Recall)
  - The formula to calculate sensitivity (recall) is:
    $$\text{Sensitivity or Recall} = \frac{\text{True Positives}}{\text{True Positives}+\text{False Negatives}}×\text{100\%}$$
  - Where:
    - True Positives (TP): The number of actual positive cases correctly identified by the model.
    - False Negatives (FN): The number of actual positive cases that were incorrectly identified as negative by the model.
- Real-Life Example
  - Consider a medical test designed to detect a serious disease:
  - Out of 100 patients who actually have the disease:
    - True Positives (TP): 80 patients were correctly identified as having the disease.
    - False Negatives (FN): 20 patients who have the disease were not detected by the test.
  - To calculate sensitivity:
    $$\text{Sensitivity} = \frac{\text{80}}{\text{80} + \text{20}}×\text{100\%}$$
    $$\text{Sensitivity} = \text{80\%}$$
  - In this case, the sensitivity of the medical test is 80%. This means that the test successfully identifies 80% of those who truly have the disease, which is critical in ensuring that patients receive timely treatment.
  - High sensitivity is particularly important in medical diagnostics to minimize missed diagnoses.

### `f1 score`

The F1-Score is a performance metric that combines both precision and sensitivity (recall) into a single score. It is particularly useful when you need to balance the trade-off between precision and recall, especially in situations where one may be more important than the other. \
The F1-Score provides a way to measure a model's accuracy by considering both false positives and false negatives.

- Formula for F1-Score

  - The formula to calculate the F1-Score is: <br />
    $$\text{F1 Score}=\text{2}×\frac{\text{Precision} × \text{Recall}}{\text{{Precision}} + \text{Recall}}$$
  - Where:
    - Precision is the proportion of true positive predictions out of all positive predictions.
    - Recall (Sensitivity) is the proportion of true positive predictions out of all actual positive cases.

- Range of F1-Score
  - The F1-Score ranges from 0 to 1.
    - 0 indicates the worst performance, meaning the model fails to make any correct positive predictions.
    - 1 represents perfect precision and recall, where the model correctly identifies all positive cases without any false positives or false negatives.
  - Generally:
    - F1-Score > 0.9: Excellent performance.
    - F1-Score between 0.8 and 0.9: Good performance.
    - F1-Score between 0.5 and 0.8: Average performance.
    - F1-Score < 0.5: Poor performance.
