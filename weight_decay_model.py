import os
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load the metadataset
metadataset_path = os.path.join("metadataset", "metadataset.csv")
metadataset = pd.read_csv(metadataset_path)

# Separate features and target
X = metadataset.drop(columns=['weight_decay'])
y = metadataset['weight_decay']

# Fixed 60:40 train:test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
model_path = os.path.join("metadataset", "weight_decay_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model, f)
print(f"Model saved to {model_path}")

# Evaluate on the test set
y_pred_test = model.predict(X_test)
test_mse = mean_squared_error(y_test, y_pred_test)
print("Mean Squared Error on the test set:", test_mse)

# Generate learning curve data
train_sizes = np.linspace(0.1, 0.9, 9)  # Now using up to 0.9
train_errors = []
test_errors = []

for train_size in train_sizes:
    X_train_part, _, y_train_part, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
    model.fit(X_train_part, y_train_part)
    
    y_train_pred = model.predict(X_train_part)
    y_test_pred = model.predict(X_test)
    
    train_errors.append(mean_squared_error(y_train_part, y_train_pred))
    test_errors.append(mean_squared_error(y_test, y_test_pred))

# Save learning curve data for later plotting
learning_curve_data = {
    "train_sizes": train_sizes,
    "train_errors": train_errors,
    "test_errors": test_errors
}

learning_curve_path = os.path.join("metadataset", "learning_curve_data.pkl")
with open(learning_curve_path, "wb") as f:
    pickle.dump(learning_curve_data, f)
print(f"Learning curve data saved to {learning_curve_path}")
