import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# Load your unbalanced dataset
data = pd.read_csv('dataset.csv')

# Separate features and labels
X = data.drop('class', axis=1)  # Replace 'class' with the actual target column name
y = data['class']

# Identify categorical columns (string) and numerical columns (float)
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['float64']).columns

# Preprocess categorical columns (e.g., apply Label Encoding)
for col in categorical_columns:
    label_encoder = LabelEncoder()
    X[col] = label_encoder.fit_transform(X[col])

# Preprocess numerical columns (e.g., apply Standardization)
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

# Initialize SMOTE and Tomek Links
smote = SMOTE(sampling_strategy='minority')
tomek = TomekLinks(sampling_strategy='all')
smt = SMOTETomek(smote=smote, tomek=tomek)

# Apply SMOTE and TOMEK Links
X_resampled, y_resampled = smt.fit_resample(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize and train an SVM classifier
svm_classifier = SVC(kernel='linear', C=1)
svm_classifier.fit(X_train[0:500], y_train[0:500])

# Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Generate a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Save the trained model to a .sav file
joblib.dump(svm_classifier, 'svm_model.sav')
