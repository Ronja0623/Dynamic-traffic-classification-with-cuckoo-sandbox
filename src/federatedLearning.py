import syft as sy
import torch
class FederatedLearning:
    def __init__(self, num_clients):
        self.num_clients = num_clients

    def set_domain(self):
        self.domain = sy.orchestra.launch(
                name="test-domain", port="auto", dev_mode=True, reset=True
        )
        return self.domain
    
    def set_clients(self):
        self.clients = [
                self.domain.login(email=f"client{i+1}@test.com", password="changethis")
                for i in range(self.num_clients)
        ]
        return self.clients
    
    def get_train_loaders(self, train_loader):
        num_per_client = len(train_loader.dataset) // self.num_clients
        train_loaders = [
            torch.utils.data.DataLoader(
                torch.utils.data.Subset(
                    train_loader.dataset,
                    range(i * num_per_client, (i + 1) * num_per_client),
                ),
                batch_size=train_loader.batch_size,
                shuffle=True,
            )
            for i in range(self.num_clients)
        ]
        return train_loaders