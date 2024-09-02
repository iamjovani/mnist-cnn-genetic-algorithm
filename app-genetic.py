import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from deap import base, creator, tools, algorithms
from scoop import futures
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import idx2numpy

# Load MNIST data
def load_mnist(path):
    images_path = f'{path}/train-images.idx3-ubyte'
    labels_path = f'{path}/train-labels.idx1-ubyte'
    test_images_path = f'{path}/t10k-images.idx3-ubyte'
    test_labels_path = f'{path}/t10k-labels.idx1-ubyte'
    train_images = idx2numpy.convert_from_file(images_path)
    train_labels = idx2numpy.convert_from_file(labels_path)
    test_images = idx2numpy.convert_from_file(test_images_path)
    test_labels = idx2numpy.convert_from_file(test_labels_path)
    return (train_images / 255.0).reshape(-1, 784), to_categorical(train_labels), (test_images / 255.0).reshape(-1, 784), to_categorical(test_labels)

# Define neural network model architecture
def create_model(num_neurons, num_layers, activation_index):
    activations = ['relu', 'sigmoid', 'tanh']
    activation = activations[activation_index]  # Convert index to string
    model = Sequential()
    model.add(Dense(num_neurons, activation=activation, input_shape=(784,)))
    for _ in range(num_layers - 1):
        model.add(Dense(num_neurons, activation=activation))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Fitness evaluation function
def evalModel(individual):
    num_neurons, num_layers, activation_index = individual
    model = create_model(num_neurons, num_layers, activation_index)
    history = model.fit(x_train, y_train, epochs=5, verbose=0, validation_split=0.1)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return (accuracy,)

# Setup for the genetic algorithm
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
toolbox.register("attr_neurons", random.randint, 64, 512)
toolbox.register("attr_layers", random.randint, 1, 5)
toolbox.register("attr_activation", random.randint, 0, 2)  # Indices for activations
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_neurons, toolbox.attr_layers, toolbox.attr_activation), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evalModel)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=[64, 1, 0], up=[512, 5, 2], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("map", futures.map)

# Load data
x_train, y_train, x_test, y_test = load_mnist('mnist')

# Run the Genetic Algorithm
population = toolbox.population(n=5)
hof = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

population, logbook = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)
generation_max_fitness = [gen['max'] for gen in logbook]

# Visualize the network evolution
plt.figure(figsize=(10, 5))
plt.plot(generation_max_fitness, label='Max Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Over Generations')
plt.legend()
plt.savefig('network_evolution.png')
plt.show()

# Train the best model found and collect history
best_model = create_model(*hof[0])
history = best_model.fit(x_train, y_train, epochs=5, verbose=1, validation_data=(x_test, y_test))

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_validation_accuracy.png')
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_validation_loss.png')
plt.show()

# Confusion Matrix
y_pred = best_model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for Best Model')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig('confusion_matrix.png')
plt.show()

# ROC Curve and AUC for multi-class classification
y_prob = best_model.predict(x_test)
fpr, tpr, roc_auc = {}, {}, {}
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for the multiclass
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta', 'lime', 'brown']
for i, color in zip(range(10), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for each digit')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png')
plt.show()

# Feature Weights Visualization
weights = best_model.layers[0].get_weights()[0]
plt.figure(figsize=(10, 8))
sns.heatmap(weights, cmap='viridis', annot=False)
plt.title('Visualization of Weights in the First Layer')
plt.xlabel('Neurons in the First Layer')
plt.ylabel('Input Features (784 pixels)')
plt.savefig('layer_weights.png')
plt.show()
