# Neural Network Pruning Project

This repository contains the code implementation of a neural network pruning project using PyTorch and scikit-learn libraries. The project aims to demonstrate the process of pruning neural networks using magnitude-based pruning and analyzing its impact on model performance.

## Libraries Used

The project utilizes the following libraries:

- **NumPy**: For numerical operations and array handling.
- **Matplotlib**: For plotting graphs and visualizations.
- **PyTorch**: For deep learning operations and model building.
- **scikit-learn**: For generating synthetic dataset for classification and splitting dataset into train and test sets.

## Project Overview

The project consists of the following components:

1. **Neural Network Model Definition**: 
    - The neural network architecture is defined using the PyTorch library. It comprises two fully connected layers with ReLU and Sigmoid activation functions.

2. **Data Generation and Preprocessing**:
    - Synthetic classification dataset is generated using scikit-learn's `make_classification` function. The dataset is split into training and testing sets while maintaining class balance and reproducibility.

3. **Model Training**:
    - The original neural network model is trained using the training dataset. The binary cross-entropy loss function and the Adam optimizer are utilized for training.

4. **Model Evaluation**:
    - The accuracy of the trained model is evaluated on the testing dataset.

5. **Magnitude-Based Pruning**:
    - A pruning method based on the magnitude of weights is applied to the trained model. The pruning threshold is varied to observe its effect on model accuracy and the number of parameters.

6. **Result Analysis**:
    - The testing accuracy and the number of parameters of the pruned models are recorded and analyzed.

## Usage

To replicate the experiment, follow these steps:

1. Install the required libraries: `numpy`, `matplotlib`, `torch`, `scikit-learn`.

2. Clone this repository:
    ```
    git clone <https://github.com/aayushmailarpwar/Model-Complexity-and-Neural-Network-Pruning-.git>
    ```

3. Navigate to the project directory:
    ```
    cd neural-network-pruning
    ```

4. Run the main script:
    ```
    python main.py
    ```

5. View the results:
    - The script generates two plots: one showing the test accuracy vs. pruning threshold and the other showing the number of parameters vs. pruning threshold.

## Results

The project demonstrates the impact of magnitude-based pruning on neural network models. By varying the pruning threshold, it is observed how the model's accuracy changes along with the reduction in the number of parameters. The results provide insights into the trade-off between model size and performance.

For more details, refer to the code implementation and the generated plots.
