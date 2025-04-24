#  FlipItNews - News Category Classification with NLP and Machine Learning

This project demonstrates an end-to-end pipeline for **multi-class text classification** using natural language processing (NLP) techniques and machine learning models. The goal is to classify news articles into predefined categories based on their content.

##  Overview

- **Dataset**: Custom dataset containing news articles and their corresponding categories.
- **Preprocessing**: Tokenization, stopword removal, lemmatization, and contraction expansion.
- **Feature Extraction**: TF-IDF and Bag-of-Words (BoW).
- **Models Trained**:
  - Naive Bayes
  - Decision Tree
  - K-Nearest Neighbors
  - Random Forest
- **Evaluation Metrics**: Accuracy, F1-Score (Macro), Classification Report, and Confusion Matrix.
- **Visualization**: Model performance comparison using bar charts.

---

##  Project Workflow

1. **Install Dependencies**

- pip install -r requirements.txt


2. **Load and Explore Dataset**

- Load `flipitnews-data.csv`
- Explore category distribution and sample articles

3. **Text Preprocessing**

- Convert to lowercase
- Expand contractions
- Remove non-letter characters
- Tokenize, remove stopwords, and lemmatize

4. **Feature Engineering**

- Use `TF-IDF` or `CountVectorizer` to convert text to numerical features

5. **Model Training & Evaluation**

- Train four classifiers
- Evaluate using:
  - Accuracy
  - F1-Score (Macro)
  - Classification Report
  - Confusion Matrix

6. **Results Visualization**

- Comparing models visually based on accuracy
![image](https://github.com/user-attachments/assets/fd416a9a-3399-4943-88f3-bc9ae43dd40f)

---

## Model Comparison Results

The following models were trained and evaluated:

| Model              | Accuracy | F1-Score (Macro) |
|-------------------|----------|------------------|
| Naive Bayes        | 0.959551   | 0.958627      |
| Random Forest      | 0.957303   | 0.957574      |
| Decision Tree      | 0.939326   | 0.939349      |
| K-Nearest Neighbors| 0.826966   | 0.827288      |

_Naive Bayes and Random Forest outperformed the other models._

---

## Technologies Used

- Python 3.x
- Pandas, NumPy
- NLTK
- Scikit-learn
- Seaborn, Matplotlib

---

##  File Structure

- flipitnews-data.csv # Dataset file 
- requirements.txt # List of dependencies 
- flipitnews_classification.py # Main Jupyter Notebook / Python script 
- README.md # Project documentation


---

## Future Improvements

- Incorporate deep learning models (e.g., LSTM, BERT)
- Handle class imbalance
- Add cross-validation for model robustness

---

## Contact

For questions or collaboration, feel free to connect:

**Vaibhav**  
Data Science & Analytics @Scaler Academy  
[LinkedIn](https://www.linkedin.com/in/vaibhav-pandey-re2103/) • [GitHub](https://github.com/vaibhpande21)

---

If you find this project useful, please consider starring the repository!

