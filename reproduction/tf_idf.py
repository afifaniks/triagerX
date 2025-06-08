# %%
import pandas as pd
import numpy as np
import pickle
import string
import re
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Ensure you have nltk resources downloaded (if not, run these lines once)
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

start_time = time.time()

# Load your dataset (replace 'your_dataset.csv' with the actual file path)
dataset = '/home/mdafifal.mamun/notebooks/triagerX/data/openj9/openj9_22112024.csv'
df = pd.read_csv(dataset)

print(f"Dataset: {dataset}")

df = df.rename(columns={"assignees": "owner", "issue_body": "description"})

# Combine issue_title and issue_body into a single feature
df['text'] = df['issue_title'] + ' ' + df['description']

data = df[df["component"].notna()]

df = df[df["component"].notna()]

classes = [
    "comp:vm",
 "comp:jit",
  "comp:jvmti",
  "comp:jitserver",
  "comp:jclextensions",
 "comp:test",
  "comp:build",
  "comp:gc",
  "comp:infra"
]



num_issues = len(df)
print(f"Total number of issues after processing: {num_issues}")

# num_cv = 10
# block = 9

# samples_per_block = len(df) // num_cv
# sliced_df = df[: samples_per_block * (block + 1)]

# print(f"Samples per block: {samples_per_block}, Selected block: {block}")

# # Train and Validation preparation

# df_train = sliced_df[: samples_per_block * block]
# df_test = sliced_df[samples_per_block * block : samples_per_block * (block + 1)]

df = df[df["component"].isin(classes)]
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, stratify=df["component"])
assert set(df_train.component.unique()) == set(df_test.component.unique())

print(f"Training data: {len(df_train)}, Validation data: {len(df_test)}")
print(f"Number of components in train: {len(df_train.component.unique())}")
print(f"Number of components in test: {len(df_test.component.unique())}")

print(f"Train dataset size: {len(df_train)}")
print(f"Test dataset size: {len(df_test)}")

# %%
data =  pd.concat([df_train, df_test], ignore_index=True, sort=False)

# %%
# Preprocessing function
def preprocess_text(text):
    # Lowercase the text
    text = str(text).lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenization (split into words)
    tokens = text.split()
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join the tokens back into a single string
    return ' '.join(tokens)

print("Cleaning text...")
# Apply preprocessing to the text data
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Define the input features (X) and the target labels (y)
X = data['cleaned_text']
y = data['component']

print("Create train test split...")
# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

print("Start TF-IDF")

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=10000)

# Fit and transform the training data, and transform the test data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)


# # %% [markdown]
# # SVM

# %%
from sklearn.svm import SVC
import numpy as np

print("Start SVM training")

# %%
# Initialize the SVM classifier
classifier = SVC(probability=True, verbose=False)  # Set probability=True to enable probability estimates

# Train the classifier
classifier.fit(X_train_tfidf, y_train)

# print("Load model")
# with open("/home/mdafifal.mamun/notebooks/triagerX/model_ts.pkl", 'rb') as file:
#     classifier = pickle.load(file)

# Make predictions
y_pred = classifier.predict(X_test_tfidf)

# # Evaluate the model
print("Overall Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print("Saving model...")

# save
# with open('model_mc.pkl','wb') as f:
#     pickle.dump(classifier,f)


batch_size = 50  # Set an appropriate batch size
decision_scores = []

for i in range(0, X_test_tfidf.shape[0], batch_size):
    print("Processing batch:", i)
    batch = X_test_tfidf[i:i + batch_size]
    decision_scores.extend(classifier.decision_function(batch))

decision_scores = np.array(decision_scores)

# %%
def top_n_accuracy_report(decision_scores, y_true, classifier, top_ns=[1, 2, 3]):
    accuracies = {}
    for n in top_ns:
        # Get the top n predicted classes for each sample
        top_n_indices = np.argsort(-decision_scores, axis=1)[:, :n]
        
        # Map indices to original class labels
        top_n_labels = [[classifier.classes_[index] for index in indices] for indices in top_n_indices]

        # Calculate accuracy based on whether true labels are in top n predictions
        accuracy = np.mean([y_true.iloc[i] in top_n_labels[i] for i in range(len(y_true))])
        accuracies[n] = accuracy
    return accuracies

# Calculate and display top 1, 3, 5, 10, 20 accuracies
top_n_accuracies = top_n_accuracy_report(decision_scores, y_test, classifier)
for n, acc in top_n_accuracies.items():
    print(f"Top-{n} Accuracy: {acc:.4f}")

time_taken = time.time() - start_time

print(f"Time taken: {time_taken} seconds")

# %%



