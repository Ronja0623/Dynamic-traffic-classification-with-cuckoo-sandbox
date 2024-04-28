import csv
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.models import VGG16_Weights
from tqdm import tqdm


class EnhancedLeNet(nn.Module):
    """
    Enhanced LeNet model with 2 convolutional layers and 2 fully connected layers.
    Cite:
        Wang, W., Zhu, M., Zeng, X., Ye, X., & Sheng, Y. (2017, January).
        Malware traffic classification using convolutional neural network
        for representation learning. In 2017 International conference on
        information networking (ICOIN) (pp. 712-717). IEEE.
    """

    def __init__(self, num_classes):
        super(EnhancedLeNet, self).__init__()
        self.image_size = (28, 28)
        # First convolutional layer C1
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=5, padding=2
        )  # padding to keep size 28x28
        # First max-pooling layer P1
        self.pool1 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Resulting in 14x14 feature maps

        # Second convolutional layer C2
        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=5, padding=2
        )  # padding to keep size 14x14
        # Second max-pooling layer P2
        self.pool2 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # Resulting in 7x7 feature maps

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Convolutional and max pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten the output for fully connected layers
        x = torch.flatten(x, 1)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        # Softmax output layer
        return F.log_softmax(x, dim=1)


class BasicCNN(nn.Module):
    """
    Basic CNN model with 2 convolutional layers and 2 fully connected layers.
    """

    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()
        self.image_size = (28, 28)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(
            32 * 7 * 7, 128
        )  # Adjust based on your input size after conv layers
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CustomVGG16(nn.Module):
    """
    VGG16 model with the last fully connected layer replaced with a new one.
    """

    def __init__(self, num_classes):
        super(CustomVGG16, self).__init__()
        self.image_size = (224, 224)
        self.model = models.vgg16(weights=VGG16_Weights.DEFAULT)
        for param in self.model.features.parameters():
            param.requires_grad = False
        self.model.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.model(x)


class DataProcessor:
    """
    Process the dataset and load it into PyTorch DataLoader.
    """

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def get_transform(self, model_image_size):
        """
        Get the transformation for the dataset.
        """
        return transforms.Compose(
            [
                transforms.Resize(model_image_size),
                transforms.ToTensor(),
            ]
        )

    def get_num_classes(self):
        """
        Get the number of classes in the dataset.
        """
        return len(os.listdir(self.dataset_dir))

    def load_data(self, batch_size, model_image_size):
        """
        Load the dataset into PyTorch DataLoader.
        """
        transform = self.get_transform(model_image_size)
        dataset = ImageFolder(self.dataset_dir, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_test_size = len(dataset) - train_size
        val_size = int(0.1 * len(dataset))
        test_size = val_test_size - val_size
        train_data, val_data, test_data = random_split(
            dataset, [train_size, val_size, test_size]
        )
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader, test_loader


class ModelTrainer:
    """
    Train and evaluate the model.
    """

    def __init__(self, model, metrics_manager, device, learning_rate, epochs):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = epochs
        self.metrics_manager = metrics_manager

    def train(self, train_loader, val_loader, model_dir):
        """
        Train the model.
        """
        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            all_preds, all_labels = [], []
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                progress_bar.set_postfix(loss=loss.item())
            # Save the model
            model_path = os.path.join(model_dir, f"model_{epoch + 1}.pth")
            torch.save(self.model.state_dict(), model_path)
            # Calculate training metrics
            train_accuracy = accuracy_score(all_labels, all_preds)
            train_precision = precision_score(
                all_labels, all_preds, average="macro", zero_division=0
            )
            train_recall = recall_score(
                all_labels, all_preds, average="macro", zero_division=0
            )
            train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            average_loss = sum(train_losses) / len(train_losses)
            # Record training metrics
            self.metrics_manager.record_metrics(
                epoch + 1,
                average_loss,
                train_accuracy,
                train_precision,
                train_recall,
                train_f1,
                "train",
            )

            # Evaluate on validation set
            self.evaluate(val_loader, epoch)

    def evaluate(self, val_loader, epoch):
        """
        Evaluate the model on the validation set.
        """
        self.model.eval()
        val_losses = []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_losses.append(loss.item())
                _, preds = torch.max(outputs.data, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            val_accuracy = accuracy_score(all_labels, all_preds)
            val_precision = precision_score(
                all_labels, all_preds, average="macro", zero_division=0
            )
            val_recall = recall_score(
                all_labels, all_preds, average="macro", zero_division=0
            )
            val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            average_loss = sum(val_losses) / len(val_losses)

            # Record validation metrics
            self.metrics_manager.record_metrics(
                epoch + 1,
                average_loss,
                val_accuracy,
                val_precision,
                val_recall,
                val_f1,
                "validate",
            )

    def test(self, test_loader):
        """
        Test the model on the test set.
        """
        self.model.eval()
        test_losses = []
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_losses.append(loss.item())
                _, preds = torch.max(outputs.data, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            test_accuracy = accuracy_score(all_labels, all_preds)
            test_precision = precision_score(
                all_labels, all_preds, average="macro", zero_division=0
            )
            test_recall = recall_score(
                all_labels, all_preds, average="macro", zero_division=0
            )
            test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            average_loss = sum(test_losses) / len(test_losses)

            # Print test results
            print(
                f"Test Loss: {average_loss}, Test Accuracy: {test_accuracy}, Test Precision: {test_precision}, Test Recall: {test_recall}, Test F1: {test_f1}"
            )


class LogManager:
    """
    Record and save the metrics.
    """

    def __init__(self):
        self.metrics = []

    def record_metrics(self, epoch, loss, accuracy, precision, recall, f1_score, phase):
        self.metrics.append(
            {
                "epoch": epoch,
                "phase": phase,
                "loss": loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }
        )

    def save_metrics(self, file_path):
        with open(file_path, "w", newline="") as file:
            fieldnames = [
                "epoch",
                "phase",
                "loss",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
            ]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for metric in self.metrics:
                writer.writerow(metric)

    def save_model_description(
        self, file_path, model, batch_size, learning_rate, epochs
    ):
        with open(file_path, "w") as file:
            file.write(f"Model: {type(model).__name__}\n")
            file.write(f"Batch Size: {batch_size}\n")
            file.write(f"Learning Rate: {learning_rate}\n")
            file.write(f"Epochs: {epochs}\n")
