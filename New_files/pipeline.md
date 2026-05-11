---
Pipeline:
    main.py
        Explanation:
            Single entry point for the full pipeline. Orchestrates all steps in order: data loading, model training, extraction, tokenization, LLM inference, and evaluation. Accepts a config file or CLI arguments to control which models, datasets, and LLMs are run.
        Functions:
            parse_args() — parses CLI arguments (config path, run flags per stage)
            load_config(path: str) — loads a YAML/JSON config defining models, datasets, LLMs, and hyperparameters
            run_pipeline(config) — executes the full pipeline end-to-end in order
            main() — entry point; calls parse_args(), load_config(), and run_pipeline()

    Data_File.py
        Explanation:
            Loads and preprocesses all datasets used in the pipeline. Provides dataset metadata for logging and reproducibility.
        Functions:
            load_dataset(name: str) — loads a single dataset by name
            preprocess(data) — normalizes, splits, and prepares data for model input
            print_data_info(data) — prints dataset statistics (nodes, edges, features, class balance)
        Relevant data calls:
            torch_geometric.datasets.EllipticBitcoinDataset
            torch_geometric.datasets.EllipticBitcoinTemporalDataset
            torch_geometric.datasets.DGraphFin

    GNN_Definition.py
        Explanation:
            Contains all GNN model class definitions and bundles them into a unified registry for easy access during training and evaluation.
        Functions/Classes:
            class GCN
            class GAT
            class GIN
            class GraphSAGE
            build_model_bundle(config: dict) — instantiates all models with shared config
    Train.py
        Explanation:
            Handles the training loop for all GNN models across all datasets. Saves trained model weights for downstream extraction.
        Functions:
            train_epoch(model, data, optimizer, criterion) — single training step
            evaluate(model, data) — computes loss and accuracy on validation/test split
            train_model(model, data, config) — full training loop with early stopping
            train_all(model_bundle, datasets, config) — trains all model–dataset combinations
            save_model(model, path) — persists trained weights to disk
            load_model(model, path) — restores weights from disk

    Extraction.py
        Explanation:
            Extracts the three core outputs from each trained GNN for a specified target node: the model prediction, the GNNExplainer explanation mask, and the node embedding with its relevant subgraph.
        Functions:
            get_prediction(model, data, target_node) — returns the GNN's class prediction for the target node
            get_explanation(model, data, target_node) — runs GNNExplainer and returns edge/feature masks
            get_embedding(model, data, target_node) — extracts the latent embedding of the target node
            get_subgraph(data, target_node, num_hops) — extracts the k-hop subgraph around the target node
            extract_all(model, data, target_node) — runs all extractions and returns a structured bundle

    LLM_Module.py
        Explanation:
            Builds textual prompts from GNN outputs (explanation masks, embeddings, subgraph structure) and sends them to local HuggingFace LLMs (Qwen, LLaMA). Uses HuggingFace's built-in AutoTokenizer and AutoModelForCausalLM internally for tokenization and generation. The GNN prediction is excluded from prompts and reserved for evaluation only.
        Functions (semantic formatting / prompt building):
            format_explanation(explanation_mask) — converts edge/feature importance scores into a human-readable explanation text
            format_embedding(embedding, max_length: int | None = None) — serializes the node embedding vector into a readable string; applies PCA reduction via reduce_embedding() if max_length is exceeded
            reduce_embedding(embedding, n_components: int) — applies PCA to compress the embedding before serialization
            format_subgraph(subgraph) — describes subgraph topology and node features in text
            build_prompt(explanation_text, embedding_text, subgraph_text, template: str) — assembles the final LLM prompt from formatted components and a prompt template
            Functions (HuggingFace LLM handling):
            load_llm(model_name: str, device: str) — loads a HuggingFace AutoTokenizer and AutoModelForCausalLM from local path or Hub and moves the model to the specified device
            generate_response(model, tokenizer, prompt: str, device: str, **gen_kwargs) — tokenizes the prompt, runs model.generate(), and decodes the output into a response string
            parse_prediction(response: str) — extracts the predicted class label from the generated response
            get_prediction_for_target(model, tokenizer, prompt: str, device: str, **gen_kwargs) — convenience wrapper: prompt → response → parsed class label
            run_inference_all(model_names: list, prompts: list, device: str) — iterates over all LLMs; for each: loads the model, runs all prompts via get_prediction_for_target(), collects results, then deletes the model and clears GPU cache before loading the next

    Evaluation.py
        Explanation:
            Compares GNN predictions against LLM predictions to compute explanation accuracy. Aggregates results across models, datasets, and LLMs.
        Functions:
            compare_predictions(gnn_pred, llm_pred) — returns match/mismatch for a single instance
            compute_accuracy(results: list) — computes accuracy score over a result set
            aggregate_results(all_results: dict) — groups and summarizes results by model, dataset, and LLM
            save_results(results, path) — persists results to CSV or JSON
            plot_results(results) — generates comparison plots across experimental dimensions
