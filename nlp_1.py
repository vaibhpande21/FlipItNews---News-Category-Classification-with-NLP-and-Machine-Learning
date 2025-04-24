# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import re
import contractions
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


# %%
data = pd.read_csv("flipitnews-data.csv")
data.head()

# %%
# Plotting the distribution of categories

plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
data["Category"].value_counts().plot(kind="bar")
plt.title("Distribution of Categories")
plt.xlabel("Category")
plt.ylabel("Count")
plt.show()

# %%
data["Article"][0]

# %%
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """Preprocess the text by removing non-letters, converting to lowercase,
    expanding contractions, removing stop words, and lemmatizing."""

    # Convert to lowercase
    text = text.lower()

    # Expand contractions
    text = contractions.fix(text)

    # Remove non-letters
    text = re.sub("[^a-zA-Z]", " ", text)

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stop words and lemmatize
    cleaned_words = [
        lemmatizer.lemmatize(words) for words in words if words not in stop_words
    ]

    return " ".join(cleaned_words)


# %%
sample_index = 100
print("Original Article:\n", data["Article"][sample_index])
print("\nProcessed Article:\n", preprocess_text(data["Article"][sample_index]))

# %%
data["Processed_Article"] = data["Article"].apply(preprocess_text)
data.head()

# %%
label_encoder = LabelEncoder()
data["Encoded_Category"] = label_encoder.fit_transform(data["Category"])

# See the mapping
label_mapping = dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
)
print("Category Encoding Mapping:\n", label_mapping)


# %%
data.head()

# %%


def vectorize_text(data, method="bow"):
    """Vectorize the text data using either Bag of Words or TF-IDF.
    Args:
        data (pd.DataFrame): DataFrame containing the text data.
        method (str): Method for vectorization ('bow' or 'tfidf').
    Returns:
        X (sparse matrix): Vectorized text data.
        vectorizer: The vectorizer used for transformation.
    """
    if method == "bow":
        vectorizer = CountVectorizer()
    elif method == "tfidf":
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Method must be either 'bow' or 'tfidf'")

    X = vectorizer.fit_transform(data["Processed_Article"])
    return X, vectorizer


# %%
X, vectorizer = vectorize_text(data, method="tfidf")  # or method='bow'
y = data["Encoded_Category"]


# %%
# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# %%
# Train a Naive Bayes model

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)

# %%
print(
    "Classification Report:\n",
    classification_report(y_test, y_pred, target_names=label_encoder.classes_),
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=label_encoder.classes_
)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Naive Bayes Confusion Matrix")
plt.show()


# %%
def train_and_evaluate_model(
    model, X_train, X_test, y_train, y_test, model_name="Model"
):
    """
    Train and evaluate a model, printing the classification report and confusion matrix.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"\n {model_name} - Classification Report")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=label_encoder.classes_
    )
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()


# %%


# 1. Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
train_and_evaluate_model(
    dt_model, X_train, X_test, y_train, y_test, model_name="Decision Tree"
)

# 2. K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
train_and_evaluate_model(
    knn_model, X_train, X_test, y_train, y_test, model_name="K-Nearest Neighbors"
)

# 3. Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_evaluate_model(
    rf_model, X_train, X_test, y_train, y_test, model_name="Random Forest"
)


# %%
def get_model_scores(model, X_train, X_test, y_train, y_test):
    """Train the model and return accuracy and F1-score."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-Score (Macro)": f1_score(y_test, y_pred, average="macro"),
    }


# %%
models = {
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}

for name, model in models.items():
    scores = get_model_scores(model, X_train, X_test, y_train, y_test)
    results[name] = scores

# Convert to DataFrame for nice display
results_df = pd.DataFrame(results).T.sort_values(by="F1-Score (Macro)", ascending=False)
print("\n Model Comparison:\n")
print(results_df)

# %%

# Reset index for plotting
results_df_plot = results_df.reset_index().rename(columns={"index": "Model"})

# Plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Accuracy bar
sns.barplot(
    data=results_df_plot, x="Model", y="Accuracy", color="skyblue", label="Accuracy"
)

plt.title("Model Comparison: Accuracy", fontsize=14)
plt.xticks(rotation=45)
plt.ylim(0, 1.05)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()
