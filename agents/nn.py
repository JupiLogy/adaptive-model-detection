import numpy as np
import torch
import torch.nn.functional as F

class vanilla_nn():
    def __init__(self, in_size=100):
        self.dtype = torch.float
        self.device = torch.device("cpu")

        self.d_in = in_size
        self.d_hidden = 10
        self.d_out = 2

        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.d_in, self.d_hidden),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.d_hidden, self.d_out),
            torch.nn.Softmax(dim=-1)
        )
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.005, momentum=0.9)

    def step(self, data):
        # Forward
        self.y_pred = self.model(torch.from_numpy(data).float())
        return self.y_pred

    def backward(self, a, o):
        # Loss
        truth = torch.tensor([0, 0])
        truth[int(o)] = 1
        self.loss = self.loss_fn(self.y_pred, truth.float())

        # Backward
        self.loss.backward()
        self.optimizer.step()
        self.model.zero_grad()
        self.optimizer.zero_grad()
