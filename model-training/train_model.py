import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("CEAS_08.csv")

# Keep only needed columns
df = df[["body", "label"]]

# Remove missing values
df = df.dropna()

# Features and target
X = df["body"]
y = df["label"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )),
    ("classifier", LogisticRegression())
])

# Train model
print("Training model...")
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, "spam_model.pkl")

print("Model saved as spam_model.pkl")