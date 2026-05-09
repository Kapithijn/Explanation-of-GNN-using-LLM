import torch

from data_utility import load_and_preprocess_data, print_data_stats
from eval_utils import evaluate_all
from GNN_def import build_model_bundle
from plot_utils import plot_all_predictions
from train_utils import set_seed, train_all
from Interpertating import get_explanation_all_models, get_target_node_embedding_all_models
from LLM_explainer import prompt_model_for_explanation, prompt_model_for_explanation_all_models 

Plots = True
Embeddings = False
Explanations = False
Llm_explainer = True
turne_on_models = ["GCN", "GAT", "GIN", "GraphSAGE"]  # Specify which models to run  ["GCN", "GAT", "GIN", "GraphSAGE"] are all the models



def run_experiment(
    dataset_path="transaction_dataset.csv",
    hidden_channels=16,
    heads=4,
    dropout=0.5,
    lr=0.005,
    weight_decay=5e-4,
    epochs=200,
    print_every=20,
    seed=42,
):
    set_seed(seed)
    graph_data, _, _, _, _ = load_and_preprocess_data(dataset_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_data = graph_data.to(device)
    print_data_stats(graph_data)

    num_classes = int(graph_data.y.max().item()) + 1
    model_bundle = build_model_bundle(
        in_channels=graph_data.x.size(1),
        out_channels=num_classes,
        hidden_channels=hidden_channels,
        heads=heads,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        device=device, models_to_include=turne_on_models
    )

    histories = train_all(model_bundle, graph_data, epochs=epochs, print_every=print_every)
    results = evaluate_all(model_bundle, graph_data)

    return graph_data, model_bundle, histories, results


def run_with_plots(**kwargs):
    graph_data, model_bundle, histories, results = run_experiment(**kwargs)
    plot_all_predictions(model_bundle, graph_data)
    return graph_data, model_bundle, histories, results

def run_without_plots(**kwargs):
    return run_experiment(**kwargs)



if __name__ == "__main__":
    if Plots:
        print("Running experiment with plots...")
        graph_data, model_bundle, histories, results = run_with_plots()
    else:
        print("Running experiment without plots...")
        graph_data, model_bundle, histories, results = run_without_plots()
    
    if Embeddings:
        print("Extracting target node embeddings...")
        target_node_idx = 0 
        embeddings = get_target_node_embedding_all_models(model_bundle, graph_data, target_node_idx)
        for model_name, emb in embeddings.items():
            print(f"{model_name} embedding for node {target_node_idx}: {emb}")
    else:
        print("Skipping embedding extraction.")

    if Explanations:
        print("Extracting explanations...")
        target_node_idx = 0
        explanations = get_explanation_all_models(model_bundle, graph_data, target_node_idx)
        for model_name, exp in explanations.items():
            print(f"{model_name} - Node feature mask for node {target_node_idx}: {exp['node_feat_mask']}")
            print(f"{model_name} - Edge mask for node {target_node_idx}: {exp['edge_mask']}")
    else:
        print("Skipping explanation extraction.")
    
    if Llm_explainer:
        target_node_idx = 0
        explanations = get_explanation_all_models(model_bundle, graph_data, target_node_idx)
        embeddings = get_target_node_embedding_all_models(model_bundle, graph_data, target_node_idx)
        print("Generating LLM explanations...")
        llm_explanations = prompt_model_for_explanation_all_models(model_bundle, graph_data, target_node_idx=target_node_idx, explanations=explanations, embeddings=embeddings)
        print("LLM Explanations:")
        for model_name, exp in llm_explanations.items():
            print(f"{model_name}: {exp}")
    else:
        print("Skipping LLM explanation generation.")
