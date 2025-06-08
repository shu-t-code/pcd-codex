import argparse
import torch
from torch import nn
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from torch_geometric.loader import DataLoader

from dataset import PointCloudEdgeDataset, read_data_list
from model import GBCNN
from utils import compute_metrics


def build_dataloader(list_file: str, batch_size: int, shuffle: bool) -> DataLoader:
    samples = read_data_list(list_file)
    dataset = PointCloudEdgeDataset(samples, k=20)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            logits = model(data)
            loss = criterion(logits, data.y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, use_amp):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            with autocast(enabled=use_amp):
                logits = model(data)
                loss = criterion(logits, data.y)
            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu())
            all_labels.append(data.y.cpu())
            total_loss += loss.item() * data.num_graphs
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics(all_labels, all_preds)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Edge classification training")
    parser.add_argument("--train-list", required=True)
    parser.add_argument("--val-list", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = build_dataloader(args.train_list, args.batch_size, shuffle=True)
    val_loader = build_dataloader(args.val_list, args.batch_size, shuffle=False)

    model = GBCNN(input_dim=6).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0], device=device))
    scaler = GradScaler(enabled=args.amp)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, args.amp)
        val_loss, metrics = evaluate(model, val_loader, criterion, device, args.amp)
        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"acc={metrics['accuracy']:.4f} prec={metrics['precision']:.4f} "
            f"rec={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
