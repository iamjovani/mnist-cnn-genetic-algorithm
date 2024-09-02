# MNIST CNN with Genetic Algorithm Optimization

This repository contains a Convolutional Neural Network (CNN) implementation for the MNIST dataset, optimized using a Genetic Algorithm (GA). The purpose of this project is to demonstrate how GAs can be used to improve neural network performance by optimizing hyperparameters such as the number of neurons, the number of layers, and the activation functions.

## Project Structure

- `mnist_cnn_ga.py`: Main script containing the CNN implementation and GA optimization.
- `requirements.txt`: List of required Python packages to run the code.
- `README.md`: Overview of the project, including setup instructions and descriptions.
- `network_evolution.png`: Plot showing the fitness evolution over generations.
- `training_validation_accuracy.png`: Plot of training and validation accuracy over epochs.
- `training_validation_loss.png`: Plot of training and validation loss over epochs.
- `confusion_matrix.png`: Confusion matrix for the best model.
- `roc_curves.png`: ROC curves for multi-class classification.
- `layer_weights.png`: Visualization of weights in the first layer.

## Getting Started

### Prerequisites

Make sure you have Python 3.7 or later installed. You can install the required Python packages using `pip` and the `requirements.txt` file provided in this repository.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/mnist-cnn-genetic-algorithm.git
    cd mnist-cnn-genetic-algorithm
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the MNIST dataset files (`.idx3-ubyte` and `.idx1-ubyte`) and place them in a directory named `mnist`.

### Running the Script

After setting up the environment, you can run the script as follows:

```bash
python mnist_cnn_ga.py
