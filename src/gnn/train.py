"""Training loop for molecular property prediction."""

import argparse

import torch
from torch_geometric.loader import DataLoader

from gnn.data import load_molecule_dataset
from gnn.model import MolecularGNN


def train(args):
    train_set, val_set = load_molecule_dataset(args.dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)
    model = MolecularGNN(
        node_dim=train_set[0].num_node_features,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_tasks=train_set.num_tasks,
    ).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = batch.cuda()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.num_graphs)
            loss = model.loss(logits, batch.y, ~torch.isnan(batch.y))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_auc = evaluate(model, val_loader)
        print(f"epoch={epoch} val_auc={val_auc:.3f}")
        best_val = min(best_val, val_auc)
    torch.save(model.state_dict(), args.checkpoint)


def evaluate(model, loader):
    model.eval()
    scores = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.cuda()
            logits = model(batch.x, batch.edge_index, batch.edge_attr, batch.num_graphs)
            scores.append(logits.sigmoid().mean().item())
    return sum(scores) / max(1, len(scores))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="tox21")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    train(parser.parse_args())
