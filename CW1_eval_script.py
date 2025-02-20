import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# Load data
training_data = pd.read_csv('cleaned_training_data.csv')
test_data = pd.read_csv('CW1_test.csv')

# Identify numerical features in the cleaned training data (excluding the target variable)
numerical_features = training_data.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('outcome')

# Apply scaling transformation
scaler = StandardScaler()
training_data[numerical_features] = scaler.fit_transform(training_data[numerical_features])
test_data[numerical_features] = scaler.transform(test_data[numerical_features])

# One-hot encoding for test dataset
categorical_cols = ['cut', 'color', 'clarity']
test_data = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)

# Align columns
test_data = test_data.reindex(columns=training_data.columns.drop('outcome'), fill_value=0)

# Separate features and target
X_train = training_data.drop(columns=['outcome'])
y_train = training_data['outcome']

# Best hyperparameters obtained from the hyperparameter tuning process
best_params = {
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 100,
    'subsample': 0.9
}

# Initialize the model with the best hyperparameters
gbr = GradientBoostingRegressor(
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    n_estimators=best_params['n_estimators'],
    subsample=best_params['subsample'],
    random_state=42
)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rfe', RFE(estimator=gbr, n_features_to_select=20)),
    ('poly', PolynomialFeatures(degree=1, include_bias=False)),
    ('model', gbr)
])

# Perform cross-validation
scores = cross_val_score(pipeline, X_train, y_train, scoring='r2', cv=5)
print("Mean R^2 with cross-validation:", scores.mean())

# Train the pipeline on the full training data
pipeline.fit(X_train, y_train)

# Generate predictions on the test data
y_test_pred = pipeline.predict(test_data)

# Save predictions
submission = pd.DataFrame({'yhat': y_test_pred})
submission.to_csv('CW1_submission_K23004648.csv', index=False)
'''
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Load data
training_data = pd.read_csv('cleaned_training_data.csv')
test_data = pd.read_csv('CW1_test.csv')

# Identify numerical features in the cleaned training data(excluding the target variable)
numerical_features = training_data.select_dtypes(include=[np.number]).columns.tolist()
numerical_features.remove('outcome') 

# Apply same scaling transformation
scaler = StandardScaler()
training_data[numerical_features] = scaler.fit_transform(training_data[numerical_features])
test_data[numerical_features] = scaler.transform(test_data[numerical_features])

# One hot encoding for test dataset
categorical_cols = ['cut', 'color', 'clarity']
test_data = pd.get_dummies(test_data, columns=categorical_cols, drop_first=True)

# Align columns
test_data = test_data.reindex(columns=training_data.columns.drop('outcome'), fill_value=0)

# Separate features and target
X_train = training_data.drop(columns=['outcome'])
y_train = training_data['outcome']

# Best hyperparameters obtained from the hyperameter tuning process 
best_params = {
    'learning_rate': 0.1,
    'max_depth': 3,
    'min_samples_leaf': 1,
    'min_samples_split': 2,
    'n_estimators': 100,
    'subsample': 0.9
}

# Initialize the model with the best hyperparameters
gbr = GradientBoostingRegressor(
    learning_rate=best_params['learning_rate'],
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    min_samples_split=best_params['min_samples_split'],
    n_estimators=best_params['n_estimators'],
    subsample=best_params['subsample'],
    random_state=42
)
# Feature Selection (RFE)
base_model = gbr
rfe = RFE(estimator=base_model, n_features_to_select=20)
rfe.fit(X_train, y_train)
X_train_rfe = rfe.transform(X_train)
test_data_rfe = rfe.transform(test_data)

# Polynomial Features
poly = PolynomialFeatures(degree=1, include_bias=False)
X_train_poly = poly.fit_transform(X_train_rfe)
test_data_poly = poly.transform(test_data_rfe)

# Model Training
model = gbr
model.fit(X_train_poly, y_train)

# Cross-validation
scores = cross_val_score(model, X_train_poly, y_train, scoring='r2', cv=5)
print("Mean R^2 with cross-validation:", scores.mean())

# Generate predictions
y_test_pred = model.predict(test_data_poly)

# Save predictions
submission = pd.DataFrame({'yhat': y_test_pred})
submission.to_csv('CW1_submission_K23004648.csv', index=False)
'''