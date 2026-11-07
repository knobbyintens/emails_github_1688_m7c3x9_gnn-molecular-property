"""Training script for clinical risk models."""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from src.model import TabularTransformer
from src.features import ClinicalDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mlhc.yaml")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = json.load(handle)

    train_set = ClinicalDataset(cfg["train_path"], feature_names=cfg.get("feature_names"))
    loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
    model = TabularTransformer(num_features=train_set.num_features, num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(cfg["epochs"]):
        model.train()
        for features, labels in loader:
            logits = model(features)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch={epoch} loss={loss.item():.4f}")
    torch.save(model.state_dict(), "outputs/tabular_transformer.pt")


if __name__ == "__main__":
    main()
