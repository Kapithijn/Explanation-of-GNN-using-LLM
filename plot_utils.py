import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_predictions(model, data, model_name="Model"):
    model.eval()
    with torch.no_grad():
        out = model(data)
        test_logits = out[data.test_mask]
        test_target = data.y[data.test_mask]

        if test_logits.size(0) == 0:
            print(f"{model_name}: no test samples to plot")
            return

        if test_logits.size(1) < 2:
            print(f"{model_name}: expected at least 2 classes for class-1 probability plot")
            return

        probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        target_np = test_target.cpu().numpy()

    plt.figure(figsize=(8, 5))
    sns.histplot(
        probs[target_np == 0],
        color="blue",
        label="Class 0",
        kde=True,
        stat="density",
        bins=30,
    )
    sns.histplot(
        probs[target_np == 1],
        color="orange",
        label="Class 1",
        kde=True,
        stat="density",
        bins=30,
    )
    plt.title(f"{model_name} Predicted Probabilities for Class 1")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.legend()
    plt.show()


def plot_all_predictions(model_bundle, data):
    for model_name, parts in model_bundle.items():
        plot_predictions(parts["model"], data, model_name=model_name)
