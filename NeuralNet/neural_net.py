import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras import backend as K
from keras.src.metrics import Metric

# Ensure TensorFlow is using GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Load the data
df = pd.read_csv('../neural_net_csv/normalized_data.csv')

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
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Assuming binary classification
])


class RegretClassification(Metric):
    def __init__(self, name='regret_classification', **kwargs):
        super(RegretClassification, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # custom_metric_values = tf.where(K.greater(y_pred, 0.5),  1 - 2 * y_true, 2 * y_true - 1)
        custom_metric_values = tf.where(K.greater(y_true, 0.5),
                                        y_true - (y_pred * y_true + (1 - y_pred) * (1 - y_true)),
                                        1 - y_true - (y_pred * y_true + (1 - y_pred) * (1 - y_true)))

        true_positives = K.sum(custom_metric_values)
        total_samples = K.cast(K.shape(y_true)[0], K.floatx())

        self.true_positives.assign_add(true_positives)
        self.total_samples.assign_add(total_samples)

    def result(self):
        return self.true_positives / self.total_samples

    def reset_states(self):
        K.batch_set_value([(v, 0) for v in self.variables])


class RegretRaw(Metric):
    def __init__(self, name='regret_raw', **kwargs):
        super(RegretRaw, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = K.cast(y_pred >= 0.5, dtype=K.floatx())
        custom_metric_values = tf.where(K.greater(y_true, 0.5),
                                        y_true - (y_pred * y_true + (1 - y_pred) * (1 - y_true)),
                                        1 - y_true - (y_pred * y_true + (1 - y_pred) * (1 - y_true)))

        true_positives = K.sum(custom_metric_values)
        total_samples = K.cast(K.shape(y_true)[0], K.floatx())

        self.true_positives.assign_add(true_positives)
        self.total_samples.assign_add(total_samples)

    def result(self):
        return self.true_positives / self.total_samples

    def reset_states(self):
        K.batch_set_value([(v, 0) for v in self.variables])


class F1ScoreClassification(Metric):
    def __init__(self, name='f1_score_classification', **kwargs):
        super(F1ScoreClassification, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.cast(y_true >= 0.5, dtype=K.floatx())
        y_pred = K.cast(y_pred >= 0.5, dtype=K.floatx())

        true_positives = K.sum(y_true * y_pred)
        false_positives = K.sum((1 - y_true) * y_pred)
        false_negatives = K.sum(y_true * (1 - y_pred))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1

    def reset_states(self):
        K.batch_set_value([(v, 0) for v in self.variables])


class F1ScoreRaw(Metric):
    def __init__(self, name='f1_score_raw', **kwargs):
        super(F1ScoreRaw, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='true_positives', initializer='zeros')
        self.false_positives = self.add_weight(name='false_positives', initializer='zeros')
        self.false_negatives = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = K.cast(y_true >= 0.5, dtype=K.floatx())

        true_positives = K.sum(y_true * y_pred)
        false_positives = K.sum((1 - y_true) * y_pred)
        false_negatives = K.sum(y_true * (1 - y_pred))

        self.true_positives.assign_add(true_positives)
        self.false_positives.assign_add(false_positives)
        self.false_negatives.assign_add(false_negatives)

    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1

    def reset_states(self):
        K.batch_set_value([(v, 0) for v in self.variables])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1ScoreRaw(), F1ScoreClassification(), RegretRaw(), RegretClassification()])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy, test_f1_raw, test_f1_classification, test_regret_raw, test_regret_classification = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
print(f"Test f1 raw: {test_f1_raw}")
print(f"Test f1 classification: {test_f1_classification}")
print(f"Test regret raw: {test_regret_raw}")
print(f"Test regret classification: {test_regret_classification}")

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
