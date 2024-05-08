import os
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np
import syft as sy
import tenseal as ts
import multiprocessing as mp

from model import ModelTrainer

# Set multiprocessing start method for compatibility on Windows
mp.set_start_method("spawn", force=True)


class FederatedModelTrainer(ModelTrainer):
    """
    A trainer for federated learning models with optional differential privacy.

    Attributes:
        model (nn.Module): The PyTorch model to train.
        metrics_manager: A manager to record and manage metrics.
        device: The device to run the model on (e.g., "cpu" or "cuda").
        learning_rate (float): The learning rate for the optimizer.
        epochs (int): The number of training epochs.
        num_clients (int): The number of clients in federated learning.
        use_federated_learning (bool): Whether to use federated learning.
        use_differential_privacy (bool): Whether to use differential privacy.
        clipping_threshold (float): The clipping threshold for differential privacy.
        granularity (float): The granularity for differential privacy.
        noise_scale (float): The scale of the Gaussian noise for differential privacy.
        rotation_type (str): The type of rotation to use ('hd' or 'dft').
        modulus (int): The modulus for modular arithmetic.
        zeroing (bool): Whether to enable adaptive zeroing for data corruption mitigation.
    """

    def __init__(
        self,
        model,
        metrics_manager,
        device,
        learning_rate,
        epochs,
        num_clients=3,
        use_federated_learning=False,
        use_differential_privacy=False,
        clipping_threshold=1.0,
        granularity=None,
        noise_scale=1.0,
        rotation_type="hd",
        modulus=None,
        zeroing=True,
    ):
        """
        Initializes the FederatedModelTrainer.

        Args:
            model (nn.Module): The PyTorch model to train.
            metrics_manager: A manager to record and manage metrics.
            device: The device to run the model on (e.g., "cpu" or "cuda").
            learning_rate (float): The learning rate for the optimizer.
            epochs (int): The number of training epochs.
            num_clients (int, optional): The number of clients in federated learning. Defaults to 1.
            use_federated_learning (bool, optional): Whether to use federated learning. Defaults to False.
            use_differential_privacy (bool, optional): Whether to use differential privacy. Defaults to False.
            clipping_threshold (float, optional): The clipping threshold for differential privacy. Defaults to None.
            granularity (float, optional): The granularity for differential privacy. Defaults to None.
            noise_scale (float, optional): The scale of the Gaussian noise for differential privacy. Defaults to 1.0.
            rotation_type (str, optional): The type of rotation to use ('hd' or 'dft'). Defaults to 'hd'.
            modulus (int, optional): The modulus for modular arithmetic. Defaults to None.
            zeroing (bool, optional): Whether to enable adaptive zeroing for data corruption mitigation. Defaults to True.
        """
        super().__init__(model, metrics_manager, device, learning_rate, epochs)
        self.num_clients = num_clients
        self.use_federated_learning = use_federated_learning
        self.use_differential_privacy = use_differential_privacy
        self.clipping_threshold = clipping_threshold
        self.granularity = granularity or 1.0  # Set default granularity to 1.0
        self.noise_scale = noise_scale
        self.rotation_type = rotation_type
        self.modulus = modulus or 2**10  # Use a default power of two for modulus
        self.zeroing = zeroing

        if self.use_federated_learning:
            self.domain = sy.orchestra.launch(name="test-domain-1", port="auto", dev_mode=True, reset=True)
            self.clients = [
                self.domain.login(email=f"client{i+1}@test.com", password="changethis")
                for i in range(self.num_clients)
            ]
        else:
            self.clients = None

    def fwht(self, x):
        """ Fast Walsh-Hadamard Transform. """
        h = 1
        while h < len(x):
            for i in range(0, len(x), h * 2):
                for j in range(i, i + h):
                    x[j], x[j + h] = x[j] + x[j + h], x[j + h] - x[j]
            h *= 2
        return x

    def secure_aggregate_client_side(self, parameters):
        """
        Secure aggregation on the client side with differential privacy.

        Args:
            parameters (generator): The model parameters to aggregate.

        Returns:
            list[np.ndarray]: The aggregated vectors with added Gaussian noise.
        """
        aggregated_vectors = []
        for param in parameters:
            vector = param.cpu().detach().numpy().flatten()
            vector_norm = np.linalg.norm(vector, 2)
            scaled_vector = (
                (1 / self.granularity)
                * min(1, self.clipping_threshold / vector_norm)
                * vector
            )

            # Pad scaled_vector to the nearest power of two
            next_power_of_two = int(2**np.ceil(np.log2(len(scaled_vector))))
            scaled_vector = np.pad(scaled_vector, (0, next_power_of_two - len(scaled_vector)))

            if self.rotation_type == "hd":
                flattened_vector = self.fwht(scaled_vector)
            elif self.rotation_type == "dft":
                flattened_vector = np.fft.fft(scaled_vector).real
            else:
                raise ValueError("Invalid rotation type. Choose 'hd' or 'dft'.")

            rounded_vector = np.round(flattened_vector).astype(int)
            noise_vector = np.round(
                np.random.normal(
                    loc=0,
                    scale=self.noise_scale / self.granularity,
                    size=len(scaled_vector),
                )
            ).astype(int)
            aggregated_vector = (rounded_vector + noise_vector) % self.modulus

            if self.zeroing:
                # Adaptive zeroing logic goes here, for now just a placeholder
                pass

            aggregated_vectors.append(aggregated_vector)
        return aggregated_vectors

    def secure_aggregate_server_side(self, aggregated_vectors):
        """
        Secure aggregation on the server side with differential privacy.

        Args:
            aggregated_vectors (list[np.ndarray]): The aggregated vectors from the clients.

        Returns:
            list[torch.Tensor]: The resulting vectors after inverse Hadamard or DFT transformation.
        """
        result_vectors = []
        for aggregated_vector in aggregated_vectors:
            adjusted_vector = (aggregated_vector - self.modulus // 2) % self.modulus - (
                self.modulus // 2
            )

            if self.rotation_type == "hd":
                result_vector = self.granularity * self.fwht(adjusted_vector)
            elif self.rotation_type == "dft":
                result_vector = self.granularity * np.fft.ifft(adjusted_vector).real
            else:
                raise ValueError("Invalid rotation type. Choose 'hd' or 'dft'.")

            result_vectors.append(torch.Tensor(result_vector).to(self.device))
        return result_vectors

    def train(self, train_loader, val_loader, model_dir):
        """
        Trains the model with optional federated learning and differential privacy.

        Args:
            train_loader (torch.utils.data.DataLoader): The training data loader.
            val_loader (torch.utils.data.DataLoader): The validation data loader.
            model_dir (str): The directory to save the model.
        """
        if self.use_federated_learning:
            # Simulate federated learning by splitting data among clients
            data_per_client = len(train_loader.dataset) // self.num_clients
            train_loaders = [
                torch.utils.data.DataLoader(
                    torch.utils.data.Subset(train_loader.dataset, range(i * data_per_client, (i + 1) * data_per_client)),
                    batch_size=train_loader.batch_size,
                    shuffle=True
                )
                for i in range(self.num_clients)
            ]
        else:
            train_loaders = [train_loader]

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            all_preds, all_labels = [], []
            z_agg = None

            total_batches = sum(len(loader) for loader in train_loaders)
            progress_bar = tqdm(total=total_batches, desc=f"Epoch {epoch + 1}", leave=True)

            for client, client_train_loader in enumerate(train_loaders):
                for batch_idx, (inputs, labels) in enumerate(client_train_loader):
                    # Send the data to the client's location
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()

                    if self.use_differential_privacy:
                        z = self.secure_aggregate_client_side(self.model.parameters())
                        if z_agg is None:
                            z_agg = z
                        else:
                            z_agg = [(za + z[i]) % self.modulus for i, za in enumerate(z_agg)]
                    else:
                        self.optimizer.step()

                    train_losses.append(loss.item())
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=loss.item(), client=client + 1, batch=batch_idx + 1)

            progress_bar.close()

            if self.use_differential_privacy and z_agg is not None:
                result = self.secure_aggregate_server_side(z_agg)
                state_dict = dict(zip(self.model.state_dict().keys(), result))
                
                # Fix: Adjust mismatched layers by ensuring same shape
                current_state_dict = self.model.state_dict()
                for key, param in state_dict.items():
                    if param.shape == current_state_dict[key].shape:
                        current_state_dict[key] = param

                self.model.load_state_dict(current_state_dict)

            # Save the model
            model_path = os.path.join(model_dir, f"model_epoch{epoch + 1}.pth")
            torch.save(self.model.state_dict(), model_path)

            # Calculate training metrics
            train_accuracy = accuracy_score(all_labels, all_preds)
            train_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
            train_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
            train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            average_loss = sum(train_losses) / len(train_losses)
            self.metrics_manager.record_metrics(epoch + 1, average_loss, train_accuracy, train_precision, train_recall, train_f1, "train")

            self.evaluate(val_loader, epoch)
