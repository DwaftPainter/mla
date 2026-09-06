"""Read-only metrics shared by validation and final test evaluation."""
import torch


def score(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, samples = 0.0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, targets).item() * targets.size(0)
            samples += targets.size(0)
            flat = (targets * num_classes + logits.argmax(dim=1)).cpu()
            confusion += torch.bincount(flat, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    if samples == 0:
        raise ValueError("Cannot evaluate an empty dataset")
    true_positive = confusion.diag().double()
    precision = true_positive / confusion.sum(0).clamp(min=1)
    recall = true_positive / confusion.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {"loss": total_loss / samples,
            "accuracy": true_positive.sum().item() / samples,
            "precision_macro": precision.mean().item(),
            "recall_macro": recall.mean().item(),
            "f1_macro": f1.mean().item(), "samples": samples,
            "confusion_matrix": confusion.tolist()}
