import torch


def evaluate(model, data, model_name="Model"):
    model.eval()
    with torch.no_grad():
        out = model(data)
        test_logits = out[data.test_mask]
        test_target = data.y[data.test_mask]

        pred = test_logits.argmax(dim=1)
        correct = (pred == test_target).sum().item()
        total = test_target.numel()
        if total > 0:
            accuracy = correct / total
        else:
            accuracy = 0.0

        result = {
            "model": model_name,
            "accuracy": accuracy,
            "num_test_samples": int(total),
        }

        print(f"{model_name} Test Accuracy: {accuracy:.4f}")

        if total > 0 and test_logits.size(1) >= 2:
            probs = torch.softmax(test_logits, dim=1)[:, 1]
            mean_prob = float(probs.mean().item())
            result["mean_class1_probability"] = mean_prob
            print(f"{model_name} Mean predicted probability for class 1: {mean_prob:.4f}")

    return result


def evaluate_all(model_bundle, data):
    results = {}
    for model_name, parts in model_bundle.items():
        results[model_name] = evaluate(parts["model"], data, model_name=model_name)
    return results
