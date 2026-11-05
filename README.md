# classification-bits
Implementing different machine learning classification algorithms

a. Problem Statement  
The objective of this project is to build, evaluate, and compare multiple supervised machine learning classification models on a multiclass dataset "Predict Students' Dropout and Academic Success" from UCI ML Repository ( URL: https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)  . By deploying an interactive dashboard using Streamlit, the system allows users to upload test data, automatically apply necessary preprocessing steps (such as feature scaling), generate predictions, and evaluate model performance across multiple standard classification metrics.

b. Dataset Description:   
The dataset integrates disparate databases from a higher education institution to profile undergraduate students across fields like agronomy, education, journalism, and technology. Features consist of baseline enrollment data (demographic, academic, and socioeconomic) and first-year academic outcomes (semesters 1 and 2). Designed for machine learning classification, the data supports models that predict student dropout and graduation. The target variable is structured as a highly imbalanced, three-category classification task. The raw data, originally formatted as a semicolon-separated CSV file, contained multi-word feature names and inconsistent quotation marks that required initial preprocessing and cleaning. Following text normalization, the dataset was partitioned into an 80:20 ratio for the training and testing sets, respectively.

c. GitHub Repository URL: https://github.com/2026aj05430/classification-bits  
The repository contains app.py for streamlit, requirements.txt for streamlit, python code for the model generation in src folder, the original dataset and the splits in data folder, and the trained model artifacts stored inside the model folder.

d. Models Used  
All the 5 models have been implemented and there performance metrics have been calculated and incorporated in the table below. The same can be viewed in   
streamlit app URL : https://classification-bitsgit-ml-student-drop.streamlit.app/  
 The repo can be cloned locally. The classifier_test.ipynb can be run local system after modifying the filepaths according to the local system path to see the evaluation metrics.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression |0.7525 |0.8901 |0.7341 |0.7525 |0.7363 |0.5932 |
| Decision Tree |0.6768 |0.7514 |0.6852 |0.6768 |0.6794 |0.4821 |
| KNN | 0.7017|0.8238 |0.6927 |0.7017 |0.6912 |0.5107 |
| Naive Bayes |0.6994 |0.8240 |0.6877 |0.6994 |0.6868 | 0.5056|
| Support Vector Machine |0.7582 |0.8911 |0.7513 |0.7582 |0.7441 |0.6063 |  

Model perforamnce observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieves a strong baseline accuracy of 0.7525 and a high AUC of 0.8901, proving to be robust and stable for linear boundaries in the dataset. |
| **Decision Tree** | Shows lower relative performance (Accuracy: 0.6768, AUC: 0.7514), likely suffering from overfitting or sensitivity to the feature space layout. |
| **KNN** | Performs moderately well (Accuracy: 0.7017, AUC: 0.8238), capturing local patterns effectively but lagging behind linear and kernelized models. |
| **Naive Bayes** | Yields consistent results (Accuracy: 0.6994, AUC: 0.8240), though the feature independence assumption limits its ceiling on this complex dataset. |
| **Support Vector Machine** | Secures the top performance across all key metrics (Accuracy: 0.7582, AUC: 0.8911, and the highest MCC of 0.6063), handling the multi-class feature distributions most effectively. |  
|Overall, Winner for the chosen dataset?|Support Vector Machine can be considerered as the overall winner |  

