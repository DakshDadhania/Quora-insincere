# Quora-insincere
 This is the repo containing all the datasets and the model related to the research paper on Sincerity of the Questions - Quora

# Quora Insincere Questions Classification Model

This repository contains a Python implementation of a deep learning model for classifying insincere questions on Quora. The model uses a combination of LSTM, attention mechanisms, and capsule networks to achieve high accuracy in identifying insincere questions. The code provided is implemented in PyTorch and can be used for training and evaluating the model.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Data Preprocessing](#data-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Hyperparameters](#hyperparameters)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Quora Insincere Questions Classification is a natural language processing (NLP) task where the goal is to identify insincere and inappropriate questions on the Quora platform. The model in this repository utilizes deep learning techniques to achieve accurate classification results.

## Installation
To use this codebase, you'll need to have Python 3.x installed along with the following libraries:

- PyTorch
- Numpy
- Pandas
- Tqdm
- Scikit-learn

You can install the required packages using pip:

```pip install torch numpy pandas tqdm scikit-learn```


## Usage
You can use this repository to perform the following tasks:

- Data Preprocessing: Prepare the Quora dataset for training the model.
- Model Training: Train the model using the provided code and hyperparameters.
- Model Evaluation: Evaluate the trained model on a validation or test set.
- Predictions: Use the trained model to predict insincerity for new questions.

## Model Architecture
The Quora Insincere Questions Classification Model is a deep learning model that uses the following key components:

- LSTM: Bidirectional Long Short-Term Memory (LSTM) layers to capture contextual information from input sequences.
- Attention: Attention mechanism to focus on important parts of the input.
- Capsule Networks: Capsule networks to learn hierarchical patterns from the data.
- Cyclic Learning Rate: Cyclic learning rate to improve convergence and generalization.

## Data Preprocessing
To prepare the Quora dataset, you can use the provided functions to build a vocabulary and tokenize the text. The data loader class will handle data batching and bucketing for efficient training.

## Training
The training process can be performed using the provided training function. You can set the hyperparameters, such as batch size, learning rate, and number of epochs, to train the model on the prepared data.

## Evaluation
After training the model, you can evaluate its performance on a validation or test set. The model will output probabilities, which can be used to determine the insincerity of questions based on a threshold.

## Hyperparameters
The hyperparameters, such as learning rate, batch size, number of epochs, and capsule network settings, can be adjusted to optimize the model's performance. The cyclic learning rate is also utilized to improve convergence.

## Contributing
Contributions to this repository are welcome! If you find any issues or have suggestions for improvements, please feel free to submit a pull request or open an issue.

## License
This project is licensed under the [MIT License](LICENSE).
