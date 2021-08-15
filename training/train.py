#!/usr/bin/env python
"""
    Script which runs the training procedure
"""
import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3)
        self.conv2 = nn.Conv2d(64, 16, 3)
        self.conv3 = nn.Conv2d(16, 16, 3)
        self.fc = nn.Linear(7744, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            progress = batch_idx / len(train_loader) * 100
            print(
                f"Train epoch {epoch} progress: {progress:.2f}% "
                f"Loss: {loss.item():.6f}"
            )


def test(model: nn.Module, device: torch.device, test_loader: DataLoader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset) * 100
    print(f"Test average loss: {test_loss:.6f} Accuracy: {accuracy:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Size of training batches",
        default=64,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs",
        default=2,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Initial learning rate for training",
        default=5e-4,
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to save trained model",
        default=os.path.join(REPO_DIR, "saved_model", "model.pt"),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to download training data",
        default=os.path.join(REPO_DIR, "data"),
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for training",
        default=0,
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    torch.manual_seed(args.seed)

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": 500}
    if use_cuda:
        kwargs = {
            "num_workers": 1,
            "pin_memory": True,
            "shuffle": True,
        }
        train_kwargs.update(kwargs)
        test_kwargs.update(kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1308,), (0.3081,))]
    )
    train_ds = datasets.MNIST(
        args.data_path, train=True, download=True, transform=transform
    )
    test_ds = datasets.MNIST(
        args.data_path, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_ds, **train_kwargs)
    test_loader = DataLoader(test_ds, **test_kwargs)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    model.eval()
    script_module = torch.jit.script(model)
    script_module.save(args.output_path)
    print(f"Model saved to {args.output_path}")


if __name__ == "__main__":
    main(parse_args())
