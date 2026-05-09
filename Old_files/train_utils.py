import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(model, optimizer, criterion, data, model_name="Model", epochs=200, print_every=20):
    """Generic training loop for one GNN model."""
    history = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data)
        logits = out[data.train_mask]
        target = data.y[data.train_mask]
        loss = criterion(logits, target)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"{model_name} loss became non-finite. Check feature preprocessing and labels."
            )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        loss_value = float(loss.item())
        history.append(loss_value)

        if (epoch + 1) % print_every == 0:
            print(f"{model_name} Epoch {epoch + 1}, Loss: {loss_value:.4f}")

    return history


def train_all(model_bundle, data, epochs=200, print_every=20):
    """Trains all models returned by build_model_bundle."""
    histories = {}
    for model_name, parts in model_bundle.items():
        histories[model_name] = train(
            parts["model"],
            parts["optimizer"],
            parts["criterion"],
            data,
            model_name=model_name,
            epochs=epochs,
            print_every=print_every,
        )
    return histories
