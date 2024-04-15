# For numerical operations and array handling
import numpy as np

# For plotting graphs and visualizations
import matplotlib.pyplot as plt

# For deep learning operations and model building
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset

# For generating synthetic dataset for classification
from sklearn.datasets import make_classification

# For splitting dataset into train and test sets
from sklearn.model_selection import train_test_split


# Define your neural network class
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First fully connected layer
        self.relu = nn.ReLU()  # ReLU activation function
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second fully connected layer
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, x):
        out = self.fc1(x)  # First linear transformation
        out = self.relu(out)  # Apply ReLU activation
        out = self.fc2(out)  # Second linear transformation
        out = self.sigmoid(out)  # Apply sigmoid activation
        return out

# Generating and Splitting the Dataset
num_samples = 2500  # Number of samples
num_features = 10  # Number of features per sample
num_classes = 2  # Number of classes
num_clusters_per_class = 2  # Number of clusters per class

# Generating synthetic classification dataset
x, y = make_classification(
    n_samples=num_samples, 
    n_features=num_features, 
    n_classes=num_classes, 
    n_clusters_per_class=num_clusters_per_class, 
    random_state=42
)

# Splitting the dataset into training and testing sets
test_size = 0.3  # Percentage of data for testing
random_state = 42  # Seed for random number generator

# Splitting the dataset while maintaining class balance and reproducibility
x_train, x_test, y_train, y_test = train_test_split(
    x, y, 
    test_size=test_size, 
    random_state=random_state
)

# Convert data to PyTorch tensors
x_train_t = torch.tensor(x_train, dtype=torch.float32)  # Convert training input data to PyTorch tensor
y_train_t = torch.tensor(y_train, dtype=torch.float32)  # Convert training target data to PyTorch tensor
x_test_t = torch.tensor(x_test, dtype=torch.float32)  # Convert testing input data to PyTorch tensor
y_test_t = torch.tensor(y_test, dtype=torch.float32)  # Convert testing target data to PyTorch tensor

# Create the model
input_size = x_train.shape[1]  # Input size based on number of features
hidden_size = 256  # Hidden layer size
output_size = 1  # Output layer size
model = Net(input_size, hidden_size, output_size)  # Instantiate the neural network model

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary cross-entropy loss function
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer with LR 0.01

# Train the model
epochs = 20  # Number of training epochs
for epoch in range(epochs):
    model.train()  # Set model to training mode
    optimizer.zero_grad()  # Reset gradients
    outputs = model(x_train_t)  # Forward pass
    loss = criterion(outputs, y_train_t.view(-1, 1))  # Calculate loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

# Evaluate the original model
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    outputs = model(x_test_t)  # Forward pass on test data
    predicted = (outputs > 0.5).float()  # Convert outputs to binary predictions
    accuracy_original = (predicted == y_test_t.view(-1, 1)).float().mean()  # Calculate accuracy
    print(f'Test Accuracy (Non Pruned Model): {accuracy_original.item()*100:.6f}%')  # Print original model accuracy

# Magnitude-based pruning
def prune_by_threshold(model, threshold):
    parameters_to_prune = (
        (model.fc1, 'weight'),  # Weight based pruning
        (model.fc2, 'weight'),  
    )
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured, #L1 Norm
        amount=threshold,
    )

# Prune the model using magnitude-based pruning with different thresholds
thresholds = np.linspace(0.01, 0.5, 20)  # Pruning thresholds
accuracies = []
num_params_list = []

for threshold in thresholds:
    # Clone the original model
    pruned_model = Net(input_size, hidden_size, output_size)  # Instantiate new model
    pruned_model.load_state_dict(model.state_dict())  # Load weights from original model

    # Prune the cloned model
    prune_by_threshold(pruned_model, threshold)

    # Count the number of parameters
    num_params = sum(p.numel() for p in pruned_model.parameters())
    
    # Evaluate the pruned model
    pruned_model.eval()  # Set pruned model to evaluation mode
    with torch.no_grad():
        outputs = pruned_model(x_test_t)  # Forward pass on test data
        predicted = (outputs > 0.5).float()  # Convert outputs to binary predictions
        accuracy = (predicted == y_test_t.view(-1, 1)).float().mean()  # Calculate accuracy
        
        print(f'Threshold: {threshold:.2f}, Test Accuracy (Pruned Model): {accuracy.item()*100:.4f}%, Number of Parameters: {num_params}')
        
        accuracies.append(accuracy.item())
        num_params_list.append(num_params)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(thresholds, accuracies, marker='o', linestyle='-')
plt.title('Test Accuracy vs. Pruning Threshold')
plt.xlabel('Pruning Threshold')
plt.ylabel('Test Accuracy')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(thresholds, num_params_list, marker='o', linestyle='-')
plt.title('Number of Parameters vs. Pruning Threshold')
plt.xlabel('Pruning Threshold')
plt.ylabel('Number of Parameters')
plt.grid(True)

plt.tight_layout()
plt.show()


