import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Disable GPU if not available
tf.config.set_visible_devices([], 'GPU')

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Input(shape=(X.shape[1],)),  # Correct way to specify input shape
    Dense(20, activation="relu"),
    Dense(10, activation="relu"),
    Dense(1)  # Output layer for regression
])

# Compile the model
model.compile(optimizer="adam", loss="mse")

# Train the model
model.fit(X_train, y_train, epochs=5)

# Save model in TensorFlow SavedModel format (Recommended for Vertex AI)
model.export("saved_model_dir")

print("Model trained and saved successfully at saved_model_dir/")
