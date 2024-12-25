## How to run the Streamlit app

- Open the integrated terminal and run the following command to install the required packages:

```bash
pip install -r requirements.txt
```

- The `requirements.txt` file contains the list of packages used in this project.
- After installing the required packages, run the following command to start the Streamlit app:

```bash
streamlit run app/navigation.py
```

- The Streamlit app will open in the default web browser.
- The app contains the following pages:
  - Performance Analyzer
  - About Us
  - Contact Us

## Packages used in this project

### `streamlit`

- Streamlit is an open-source app framework for Machine Learning and Data Science projects.
- It is used to create web applications with Python.
- Streamlit is a great tool for prototyping and building simple data apps.
- Read the documentation from [here](https://docs.streamlit.io/).

### `pandas`

- Pandas is a fast, powerful, flexible, and easy-to-use open-source data manipulation and data analysis library built on top of the Python programming language.
- It is used to clean, transform, and analyze data.
- Read the documentation from [here](https://pandas.pydata.org/docs/user_guide/index.html).

### `scikit-learn`

- scikit-learn is a free machine learning library for Python.
- It features various algorithms like support vector machine, random forests, k-neighbors etc.
- scikit-learn is also called as sklearn.
- Read the documentation from [here](https://scikit-learn.org/stable/supervised_learning.html).

### `seaborn`

- seaborn is a Python data visualization library based on matplotlib.
- It provides a high-level interface for drawing attractive and informative statistical graphics.
- Read the documentation from [here](https://seaborn.pydata.org/).

### `matplotlib.pyplot`

- matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
- matplotlib.pyplot is a collection of command style functions that make matplotlib work like MATLAB.
- Read the documentation from [here](https://matplotlib.org/stable/api/pyplot_summary.html#module-matplotlib.pyplot).

### `io`

- The io module provides Python’s main facilities for dealing with various types of I/O.
- There are three main types of I/O: text I/O, binary I/O, and raw I/O.
- Read the documentation from [here](https://docs.python.org/3/library/io.html#io.StringIO).

### `warnings`

- The warnings module is used to handle warnings in Python.
- Warning messages are typically issued in situations where it is useful to alert the user of some condition in a program, where that condition (normally) doesn’t warrant raising an exception and terminating the program.
- Read the documentation from [here](https://docs.python.org/3/library/warnings.html#warnings.filterwarnings).
