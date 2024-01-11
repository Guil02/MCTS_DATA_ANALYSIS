import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Ensure TensorFlow is using GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the data
df = pd.read_csv(r'C:\Users\mjhri\PycharmProjects\MCTS_DATA_ANALYSIS\NeuralNet\neural_net_csv\normalized_data.csv')

# Include game concepts and pair by ID
# Assuming columns like 'GameConcept1', 'GameConcept2', etc., and 'GameID'
game_concepts_and_agents = df.columns[(df.columns != 'GameRulesetName') & (df.columns != 'Id') &
                                      (df.columns != 'Losses') & (df.columns != 'Draws') &
                                      (df.columns != 'Wins') & (df.columns != 'Games') &
                                      (df.columns != 'Win_Chance')]
X = df[game_concepts_and_agents]  # Add other relevant features
y = df['Win_Chance']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train.shape[1])

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Assuming binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Get the weights of the first layer
weights, biases = model.layers[0].get_weights()

# Calculate the magnitude of the weights
weights_magnitude = np.sum(np.abs(weights), axis=1)

# Get the indices of the weights in descending order of magnitude
sorted_indices = np.argsort(weights_magnitude)[::-1]

# Display the top N weights with their indices and values
N = 30  # Number of top weights to display
for i in range(N):
    index = sorted_indices[i]
    value = weights_magnitude[index]
    print(f'Input {game_concepts_and_agents[index]} has a weight magnitude of {value}')

# Function to plot accuracy over epochs
def plot_accuracy(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

# Example usage
plot_accuracy(history)

# Predict on new data
# new_data = ...  # replace with new data
# new_data_scaled = scaler.transform(new_data)
# predictions = model.predict(new_data_scaled)