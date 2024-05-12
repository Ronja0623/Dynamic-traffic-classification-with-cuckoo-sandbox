import multiprocessing as mp
import os

import numpy as np
import syft as sy
import tenseal as ts
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

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
        use_homomorphic_encryption=False,
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
            use_homomorphic_encryption (bool, optional): Whether to use homomorphic encryption. Defaults to False.
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
        self.use_homomorphic_encryption = use_homomorphic_encryption
        self.clipping_threshold = clipping_threshold
        self.granularity = granularity or 1.0  # Default granularity
        self.noise_scale = noise_scale
        self.rotation_type = rotation_type
        self.modulus = modulus or 2**10  # Default modulus
        self.zeroing = zeroing

        # Federated learning setup
        if self.use_federated_learning:
            self.domain = sy.orchestra.launch(
                name="test-domain-1", port="auto", dev_mode=True, reset=True
            )
            self.clients = [
                self.domain.login(email=f"client{i+1}@test.com", password="changethis")
                for i in range(self.num_clients)
            ]

        # Homomorphic encryption setup
        if self.use_homomorphic_encryption:
            self.setup_homomorphic_encryption()

    def setup_homomorphic_encryption(self):
        """
        Sets up the homomorphic encryption context for secure aggregation.
        """
        context = ts.Context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        self.context = context

    def fwht(self, x):
        """Fast Walsh-Hadamard Transform."""
        h = 1
        while h < len(x):
            for i in range(0, len(x), h * 2):
                for j in range(i, i + h):
                    x[j], x[j + h] = x[j] + x[j + h], x[j + h] - x[j]
            h *= 2
        return x

    def secure_aggregate_with_dp(self, parameters):
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
            next_power_of_two = int(2 ** np.ceil(np.log2(len(scaled_vector))))
            scaled_vector = np.pad(
                scaled_vector, (0, next_power_of_two - len(scaled_vector))
            )

            if self.rotation_type == "hd":
                flattened_vector = self.fwht(scaled_vector)
            elif self.rotation_type == "dft":
                flattened_vector = np.fft.fft(scaled_vector).real
            else:
                raise ValueError("Invalid rotation type. Choose 'hd' or 'dft'.")

            rounded_vector = np.round(flattened_vector).astype(int)
            # Add Gaussian noise to the rounded vector
            noise_vector = np.round(
                np.random.normal(
                    loc=0,
                    scale=self.noise_scale / self.granularity,
                    size=len(scaled_vector),
                )
            ).astype(int)
            aggregated_vector = (rounded_vector + noise_vector) % self.modulus
            # TODO: Adaptive zeroing logic goes here
            if self.zeroing:
                pass
            aggregated_vectors.append(aggregated_vector)
        return aggregated_vectors

    def secure_aggregate_with_he(self, parameters):
        """
        Secure aggregation on the client side with homomorphic encryption.

        Args:
            parameters (generator): The model parameters to aggregate.
        Returns:
            ts.CKKSVector: The encrypted aggregated vector.
        """
        # Create TenSEAL context inside the function
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        # Extract parameters and convert to flattened numpy arrays
        params_list = [param.cpu().detach().numpy().flatten() for param in parameters]
        # Determine the maximum size across all parameters
        max_size = max(param.size for param in params_list)
        # Pad each parameter to the maximum size and encrypt
        encrypted_vectors = []
        for param in params_list:
            if param.size < max_size:
                # Pad with zeros
                padded_param = np.pad(
                    param, (0, max_size - param.size), "constant", constant_values=0
                )
            else:
                padded_param = param
            # Convert to CKKSVector and add to the list
            encrypted_vector = ts.ckks_vector(context, padded_param.tolist())
            encrypted_vectors.append(encrypted_vector)

        return encrypted_vectors

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
        Trains the model with optional federated learning, differential privacy, and homomorphic encryption.

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
                    torch.utils.data.Subset(
                        train_loader.dataset,
                        range(i * data_per_client, (i + 1) * data_per_client),
                    ),
                    batch_size=train_loader.batch_size,
                    shuffle=True,
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
            progress_bar = tqdm(
                total=total_batches, desc=f"Epoch {epoch + 1}", leave=True
            )

            for client, client_train_loader in enumerate(train_loaders):
                for batch_idx, (inputs, labels) in enumerate(client_train_loader):
                    # Send the data to the client's location
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()

                    if self.use_differential_privacy:
                        dp_params = self.secure_aggregate_with_dp(
                            self.model.parameters()
                        )
                        if self.use_homomorphic_encryption:
                            # Encrypt the differentially private parameters
                            he_params = self.secure_aggregate_with_he(dp_params)
                            if z_agg is None:
                                z_agg = he_params
                            else:
                                z_agg = [
                                    he_agg + he for he_agg, he in zip(z_agg, he_params)
                                ]
                        else:
                            if z_agg is None:
                                z_agg = dp_params
                            else:
                                z_agg = [za + dp for za, dp in zip(z_agg, dp_params)]
                    elif self.use_homomorphic_encryption:
                        he_params = self.secure_aggregate_with_he(
                            self.model.parameters()
                        )
                        if z_agg is None:
                            z_agg = he_params
                        else:
                            z_agg = [
                                he_agg + he for he_agg, he in zip(z_agg, he_params)
                            ]
                    else:
                        self.optimizer.step()

                    train_losses.append(loss.item())
                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    progress_bar.update(1)
                    progress_bar.set_postfix(
                        loss=loss.item(), client=client + 1, batch=batch_idx + 1
                    )

            progress_bar.close()

            # Aggregate and update the model parameters at the end of each epoch
            if z_agg is not None:
                # Assuming decryption and aggregation are required
                if self.use_homomorphic_encryption:
                    # Decrypt the results before updating model parameters
                    decrypted_results = [
                        self.decrypt_aggregated_updates(vec) for vec in z_agg
                    ]
                    result = decrypted_results
                else:
                    result = z_agg

                # Fix mismatched layers by ensuring the same shape
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
            train_precision = precision_score(
                all_labels, all_preds, average="macro", zero_division=0
            )
            train_recall = recall_score(
                all_labels, all_preds, average="macro", zero_division=0
            )
            train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            average_loss = sum(train_losses) / len(train_losses)
            self.metrics_manager.record_metrics(
                epoch + 1,
                average_loss,
                train_accuracy,
                train_precision,
                train_recall,
                train_f1,
                "train",
            )

            self.evaluate(val_loader, epoch)
