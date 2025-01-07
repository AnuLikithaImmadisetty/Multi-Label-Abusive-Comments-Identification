# Multi Label Abusive Comments Identification  
Welcome to the "Multi-Label Abusive Comments Identification" repository! This project is dedicated to detecting and categorizing abusive comments using machine learning and natural language processing techniques. It leverages a combination of traditional algorithms and deep learning models to ensure accurate and effective identification of various forms of toxic language.

## Table of Contents   
 - [Overview](#overview)
 - [Project Structure](#project-structure)
 - [Dataset and Files Description](#dataset-and-files-description)
 - [Features](#features)
 - [Tools and Technologies](#tools-and-technologies)
 - [Documentation](#documentation)
 - [Getting Started](#getting-started)
 - [Usage](#usage)
 - [Contributing](#contributing)
 - [License](#license)

## Overview  
The Multi-Label Abusive Comments Identification project is designed to offer a robust solution for detecting and classifying abusive language in comments. Through the application of machine learning as well as deep learning techniques, the project effectively identifies multiple forms of toxic behavior, providing critical insights for enhancing online interactions and content moderation strategies.

## Project Structure 
Here's an overview of the project's structure
```bash
│ 
├──Datasets/        
│    ├── train.csv 
│    ├── test.csv 
│    ├── sample_submissions.csv
│    └── test_labels.csv  
│
├── Documentations/
│   ├── AMD-III-36.PPT
│   ├── Metrics.pdf
│   └── OCIT Research Paper Publish - FINAL.pdf
│
├── Images/            
│   ├── TF-IDF Plots
│   └── Word2Vec Plots      
│
└── Models/           
    ├── 1 - Logistic Regression     
    │       ├── LR (TF-IDF).ipynb
    │       ├── LR (Word2Vec).ipynb
    │       ├── submission.csv
    │       └── word_vectors.kv
    ├── 2 - Naive Bayes     
    │       ├── NB (TF-IDF).ipynb
    │       ├── NB (Word2Vec).ipynb
    │       ├── submission.csv
    │       └── word_vectors.kv
    ├── 3 - Ada Boost     
    │       ├── AB (TF-IDF).ipynb
    │       ├── AB (Word2Vec).ipynb
    │       ├── submission.csv
    │       └── word_vectors.kv
    ├── 4 - Gradient Boosting    
    │       ├── GB (TF-IDF).ipynb
    │       ├── GB (Word2Vec).ipynb
    │       ├── submission.csv
    │       └── word_vectors.kv
    ├── 5 - LSTM     
    │       ├── LSTM (TF-IDF).ipynb
    │       ├── LSTM (Word2Vec).ipynb
    │       ├── submission.csv
    │       └── word_vectors.kv
    └── 6 - BI-LSTM     
            ├── BI-LSTM (TF-IDF).ipynb
            ├── BI-LSTM (Word2Vec).ipynb
            ├── submission.csv
            └── word_vectors.kv  
```

## Dataset and Files Description 
The dataset features Wikipedia comments that have been annotated by human raters for various forms of toxic behavior. The goal is to develop a model that predicts the likelihood of each type of toxicity for every comment. The types of toxicity are:   
**_(Toxic,_** **_Severe Toxic,_** **_Obscene,_** **_Threat,_** **_Insult and_** **_Identity Hate)_**  

Here are the brief descriptions of the dataset files:    
**train.csv:** Contains the training dataset with comments and corresponding binary labels indicating toxicity.  
**test.csv:**  Provides the test dataset for which toxicity probabilities are to be predicted; some comments are excluded from scoring to avoid manual labeling.  
**sample_submission.csv:** A template file illustrating the required format for submission predictions.  
**test_labels.csv:** Includes labels for the test set, with a value of -1 indicating comments not included in the scoring process.

Refer to the [Datasets](https://github.com/AnuLikithaImmadisetty/Multi-Label-Abusive-Comments-Identification/tree/main/Datasets) here!

## Features 
The Multi-Label Abusive Comments Identification project incorporates the following components:  
1. **Text Preprocessing:** Employs advanced techniques for cleaning and preparing text data, including tokenization, stop-word removal, and stemming, to enhance data quality.
2. **Feature Extraction:** Converts text into numerical representations using techniques like TF-IDF and Word2Vec facilitating effective model input.
3. **Multi-Label Classification:** Categorizes comments into multiple abusive categories, including toxic, severe_toxic, obscene, threat, insult, and identity_hate.
4. **Model Training:** Utilizes a variety of machine learning and deep learning models, such as Logistic Regression, Naive Bayes, Ada Boost, Gradient Boosting, LSTM, and Bi-LSTM, to train the classification model.
5. **Performance Evaluation:** Evaluates model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC to ensure accurate and reliable predictions.
6. **Visualization:** Provides visualizations of results and model performance for better insight and analysis.

## Tools and Technologies 
This project leverages a variety of machine learning and deep learning algorithms to identify abusive comments. The models used include:    
### Machine Learning & Deep Learning Models:   
🔍**Logistic Regression:** For baseline classification using TF-IDF and Word2Vec.  
🔍**Naive Bayes:** Probabilistic classification with TF-IDF and Word2Vec.  
🔍**Ada Boost:** Boosting techniques with TF-IDF and Word2Vec to improve performance.  
🔍**Gradient Boosting:** Advanced boosting with TF-IDF and Word2Vec.  
🧠**LSTM (Long Short-Term Memory):** Sequence prediction using TF-IDF and Word2Vec embeddings.  
🧠**Bi-LSTM (Bidirectional LSTM):** Enhances sequence prediction by processing text in both directions with TF-IDF and Word2Vec embeddings.

### Additional Technologies:  
🤖🧠**TensorFlow/Keras:** For implementing LSTM and Bi-LSTM models.  
📊🧩**Scikit-Learn:** For various machine learning models and data preprocessing.  
📚🗣️**NLTK/Spacy:** For natural language processing tasks.  
🐼🔢**Pandas/Numpy:** For data manipulation and numerical operations.   
📉📈**Matplotlib/Seaborn:** For data visualization and model performance analysis.   
💻📓**Jupyter Notebook/Google Colab:** For interactive model development and testing.

Refer to the [Models](https://github.com/AnuLikithaImmadisetty/Multi-Label-Abusive-Comments-Identification/tree/main/Models) here!

## Documentation  
Refer to the [Research Paper]([https://github.com/AnuLikithaImmadisetty/Multi-Label-Abusive-Comments-Identification/blob/main/Documentations/OCIT Research Paper Publish - FINAL.pdf](https://github.com/AnuLikhithaImmadisetty/Multi-Label-Abusive-Comments-Identification/blob/main/Documentations/OCIT%20Research%20Paper%20Publish%20-%20FINAL.pdf)) here!

## Getting Started  
To get started with the Multi Label Abusive Comments Identification project, follow these steps:  
1. **Clone the Repository:** Clone the repository to your local machine using Git: (`git clone https://github.com/AnuLikithaImmadisetty/Multi-Label-Abusive-Comments-Identification.git`)
2. **Navigate to the Project Directory:** Change to the project directory: (`cd Multi-Label-Abusive-Comments-Identification`)
3. **Install Dependencies:** Install the required dependencies using pip: (`pip install -r requirements.txt`)
4. **Prepare the Data:** Download the dataset and place it in the `/Datasets` folder and ensure the data is in the expected format as described in the data documentation.
5. **Run the Analysis:** Open the Jupyter notebooks in Google Collab or Visual Studio Code located in the `/Models` folder, here you can explore various models and run the corresponding scripts to process the data, train the models, and make predictions.
6. **Evaluate the Models:** Review the evaluation metrics and results in the `/Models` folder.Metrics are stored in the 'metrics_summary.csv' & 'metrics_summary_alogorithmnames.csv' files and analyze the performance of the models and compare their predictions, as well fine-tune the models if necessary.
  
## Usage 
To use the trained models for Multi-Label Abusive Comments Identification:  
1. Format and preprocess your text data to align with the training data specifications.
2. Utilize the provided scripts or notebooks in the `/Models` directory to load the trained model and generate predictions.
3. Examine the output to identify which abusive categories such as (toxic, severe_toxic, obscene, threat, insult, and identity_hate) are applicable to each comment.

## Contributing 
Contributions are welcome to this project. To contribute, please follow these steps:  
1. Fork the Repository.
2. Create a New Branch (`git checkout -b feature/YourFeature`).
3. Make Your Changes.
4. Commit Your Changes (`git commit -m 'Add new feature'`).
5. Push to the Branch (`git push origin feature/YourFeature`).
6. Create a Pull Request.

## License 
This project is licensed under the MIT License. Refer to the LICENSE file included in the repository for details.
