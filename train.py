import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, make_scorer

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Define a custom multi-output accuracy scorer
def multioutput_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = []
    for i in range(y_true.shape[1]):
        scores.append(accuracy_score(y_true[:, i], y_pred[:, i]))
    return np.mean(scores)

multioutput_accuracy_scorer = make_scorer(multioutput_accuracy)

# -------------------------------
# Load and prepare your dataset
# -------------------------------
# Update the filename/path as needed.
df = pd.read_excel('Tickets_5.xlsx')

# Ensure the required columns exist: Description, Subject, Category, Sub-Category, Group
df = df[['Description', 'Subject', 'Category', 'Sub-Category', 'Group']]
df.dropna(subset=['Description', 'Subject', 'Category', 'Sub-Category', 'Group'], inplace=True)

# Combine text fields into one feature (you can include more fields if needed)
df['combined_text'] = df['Description'] + " " + df['Subject']

# Define features and targets
X = df['combined_text']
y = df[['Category', 'Sub-Category', 'Group']]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# Build the pipeline and perform GridSearchCV
# -------------------------------
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stopwords.words('english'))),
    ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000)))
])

parameters = {
    'tfidf__max_df': [0.75, 1.0],
    'tfidf__min_df': [1, 5],
    'clf__estimator__C': [0.1, 1, 10]
}

grid_search = GridSearchCV(pipeline, param_grid=parameters, cv=3,
                           scoring=multioutput_accuracy_scorer,
                           verbose=1, n_jobs=-1)

grid_search.fit(X_train, y_train)
print("Best parameters found:", grid_search.best_params_)

# Use the best estimator from grid search
best_model = grid_search.best_estimator_

# -------------------------------
# Evaluate the model
# -------------------------------
y_pred = best_model.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns)

for col in y_test.columns:
    print(f"\nClassification Report for {col}:")
    print(classification_report(y_test[col], y_pred_df[col]))

# -------------------------------
# Save the trained model
# -------------------------------
joblib.dump(best_model, 'ticket_classifier_model.pkl')
print("Model saved as ticket_classifier_model1.pkl")
