import syft as sy
import torch


class FederatedLearning:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.domain = sy.orchestra.launch(
            name="test-domain-1", port="auto", dev_mode=True, reset=True
        )
        self.clients = [
            self.domain.login(email=f"client{i+1}@test.com", password="changethis")
            for i in range(self.num_clients)
        ]

    def get_train_loaders(self, train_loader):
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
        return train_loaders
