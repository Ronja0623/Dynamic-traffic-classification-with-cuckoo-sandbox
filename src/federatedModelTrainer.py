import os

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from differentialPrivacy import DifferentialPrivacy
from federatedLearning import FederatedLearning
from homomorphicEncryption import HomomorphicEncryption
from model import ModelTrainer


class FederatedModelTrainer(ModelTrainer):
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
        granularity=1.0,
        noise_scale=1.0,
        rotation_type="hd",
        modulus=None,
        zeroing=True,
    ):
        super().__init__(model, metrics_manager, device, learning_rate, epochs)
        self.use_federated_learning = use_federated_learning
        self.use_differential_privacy = use_differential_privacy
        self.use_homomorphic_encryption = use_homomorphic_encryption

        # Federated learning setup
        if self.use_federated_learning:
            self.federated_learning = FederatedLearning(num_clients)
            self.domain = self.federated_learning.domain
            self.clients = self.federated_learning.clients

        # Differential privacy setup
        if self.use_differential_privacy:
            self.differential_privacy = DifferentialPrivacy(
                granularity=granularity,
                clipping_threshold=clipping_threshold,
                noise_scale=noise_scale,
                modulus=modulus or 2**10,
                rotation_type=rotation_type,
                zeroing=zeroing,
            )

        # Homomorphic encryption setup
        if self.use_homomorphic_encryption:
            self.homomorphic_encryption = HomomorphicEncryption()
            self.context = self.homomorphic_encryption.context

    def train(self, train_loader, val_loader, model_dir):
        if self.use_federated_learning:
            train_loaders = self.federated_learning.get_train_loaders(train_loader)
        else:
            train_loaders = [train_loader]

        z_agg = None

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            all_preds, all_labels = [], []

            total_batches = sum(len(loader) for loader in train_loaders)
            progress_bar = tqdm(
                total=total_batches, desc=f"Epoch {epoch + 1}", leave=True
            )

            for client, client_train_loader in enumerate(train_loaders):
                for batch_idx, (inputs, labels) in enumerate(client_train_loader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()

                    if self.use_differential_privacy:
                        dp_params = self.differential_privacy.process_parameters(
                            self.model.parameters()
                        )
                        if self.use_homomorphic_encryption:
                            he_params = self.homomorphic_encryption.encrypt(dp_params)
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
                        he_params = self.homomorphic_encryption.encrypt(
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

            if self.use_differential_privacy and z_agg is not None:
                if self.use_homomorphic_encryption:
                    decrypted_results = [
                        self.homomorphic_encryption.decrypt([vec])[0] for vec in z_agg
                    ]
                    result = decrypted_results
                else:
                    result = z_agg

                state_dict = dict(zip(self.model.state_dict().keys(), result))
                current_state_dict = self.model.state_dict()
                for key, param in state_dict.items():
                    if param.shape == current_state_dict[key].shape:
                        current_state_dict[key] = param

                self.model.load_state_dict(current_state_dict)

            model_path = os.path.join(model_dir, f"model_epoch{epoch + 1}.pth")
            torch.save(self.model.state_dict(), model_path)

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
