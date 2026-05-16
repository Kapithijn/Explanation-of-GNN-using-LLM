from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from sklearn.decomposition import PCA
from typing import Tuple, Dict, List, Optional, Any
import re
Top_K = 5  # Number of top edges/features to include in explanations
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"



def format_explanation(explanation_mask):
    """
    Convert edge/feature importance scores into human-readable explanation text.
    
    Args:
        explanation_mask: Edge or feature importance scores (e.g., from GNNExplainer)
    
    Returns:
        str: Human-readable explanation text describing important edges/features
    """

    sorted_scores, sorted_indices = torch.sort(explanation_mask, descending=True)
    top_indices = sorted_indices[:Top_K]
    top_scores = sorted_scores[:Top_K]
    explanation_text = "Top important edges/features:\n"
    for idx, score in zip(top_indices, top_scores):
        explanation_text += f" - Index {idx.item()} with importance {score.item():.4f}\n"
    return explanation_text



def reduce_embedding(embedding, n_components: int):
    """
    Apply PCA to compress an embedding vector.
    
    Args:
        embedding: Node embedding vector (numpy array or torch tensor)
        n_components: Number of components to reduce to
    
    Returns:
        np.ndarray: Compressed embedding
    """
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.cpu().numpy()
    
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    
    pca = PCA(n_components=n_components)
    reduced_embedding = pca.fit_transform(embedding)
    return reduced_embedding


def format_embedding(embedding, max_length: int | None = None):
    """
    Serialize node embedding vector into a readable string.
    Optionally apply PCA reduction if max_length is exceeded.
    
    Args:
        embedding: Node embedding vector (numpy array or torch tensor)
        max_length: Optional max number of components to keep (triggers PCA if needed)
    
    Returns:
        str: String representation of the embedding
    """
    embedding_size = embedding.numel() if isinstance(embedding, torch.Tensor) else embedding.size
    if max_length is not None and embedding_size > max_length:
        embedding = reduce_embedding(embedding, n_components=max_length)
    embedding_text = "embedding: ["
    for embed in embedding.flatten():
        embedding_text += f"{embed:.4f}, "
    embedding_text = embedding_text.rstrip(", ") + "]"
    return embedding_text


def format_subgraph(subgraph):
    """
    Describe subgraph topology and node features in text.
    
    Args:
        subgraph: Subgraph data (torch_geometric.Data object or similar)
    
    Returns:
        str: Description of nodes, edges, and features in the subgraph
    """
    node_features = subgraph.x if hasattr(subgraph, 'x') else None
    num_nodes = subgraph.num_nodes if hasattr(subgraph, 'num_nodes') else "unknown"
    num_edges = subgraph.num_edges if hasattr(subgraph, 'num_edges') else "unknown"

    if node_features is not None:
        feature_dim = node_features.shape[1]
        subgraph_text = f"Subgraph with {num_nodes} nodes, {num_edges} edges. Node features: {feature_dim}-dim"
    else:
        subgraph_text = f"Subgraph with {num_nodes} nodes, {num_edges} edges. Node features: unknown"

    return subgraph_text


def build_prompt(explanation_text: str, embedding_text: str, subgraph_text: str, template: str):
    """
    Assemble final LLM prompt from formatted components and a prompt template.
    
    Args:
        explanation_text: Formatted explanation from format_explanation()
        embedding_text: Formatted embedding from format_embedding()
        subgraph_text: Formatted subgraph from format_subgraph()
        template: Prompt template with placeholders like {explanation}, {embedding}, {subgraph}
    
    Returns:
        str: Complete prompt ready for LLM inference
    """
    prompt = template.format(explanation=explanation_text, embedding=embedding_text, subgraph=subgraph_text)
    prompt += "\nReturn the predicted class in the following format: 'The predicted class is X' where X is the class label or index. Select for X (0 or 1) 0 for licit and 1 for illicit." 

    return prompt



def load_llm(
    model_name: str,
    device: str,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
):
    """Load a HuggingFace LLM (AutoTokenizer and AutoModelForCausalLM).

    Notes:
        - Downloading/caching avoids repeated network downloads, but the model must
          still be loaded into RAM (and possibly GPU/MPS memory) for inference.
        - On macOS/MPS, loading a multi-billion-parameter model in float32 can easily
          exceed available unified memory. We default to float16 on MPS/CUDA.

    Args:
        model_name: Model name or local path.
        device: "cuda" | "mps" | "cpu".
        cache_dir: Optional Hugging Face cache directory.
        local_files_only: If True, never hit the network (requires files in cache).

    Returns:
        (tokenizer, model)
    """

    tokenizer_kwargs: Dict[str, Any] = {}
    if cache_dir is not None:
        tokenizer_kwargs["cache_dir"] = cache_dir
    if local_files_only:
        tokenizer_kwargs["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32

    model_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if cache_dir is not None:
        model_kwargs["cache_dir"] = cache_dir
    if local_files_only:
        model_kwargs["local_files_only"] = True

    # Try low-memory loading when available; fall back to standard loading if the
    # current transformers/accelerate combo doesn't support it.
    try:
        model: Any = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            **model_kwargs,
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    model.to(device)
    model.eval()
    return tokenizer, model


def generate_response(model, tokenizer, prompt: str, device: str, **gen_kwargs):
    """
    Tokenize prompt, run LLM generation, and decode output.
    
    Args:
        model: Loaded AutoModelForCausalLM model
        tokenizer: Loaded AutoTokenizer
        prompt: Input prompt string
        device: Device model is on
        **gen_kwargs: Additional kwargs for model.generate() (e.g., max_new_tokens=50)
    
    Returns:
        str: Generated response text (decoded output)
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    generation_kwargs = dict(gen_kwargs)
    if "max_new_tokens" not in generation_kwargs and "max_length" not in generation_kwargs:
        generation_kwargs["max_new_tokens"] = 64
    generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)

    # Many instruct models ship a generation_config with sampling enabled.
    # For this pipeline we prefer deterministic outputs (and it avoids MPS/FP16
    # numerical issues that can yield NaN/Inf probabilities during sampling).
    generation_kwargs.setdefault("do_sample", False)

    # Basic sanity for common sampling params if the user explicitly enables sampling.
    if bool(generation_kwargs.get("do_sample")):
        temperature = generation_kwargs.get("temperature")
        if temperature is None:
            generation_kwargs["temperature"] = 1.0
        else:
            try:
                temperature_value = float(temperature)
            except Exception:
                temperature_value = 1.0
            if temperature_value <= 0:
                generation_kwargs["temperature"] = 1.0

        top_p = generation_kwargs.get("top_p")
        if top_p is not None:
            try:
                top_p_value = float(top_p)
            except Exception:
                top_p_value = 1.0
            if not (0.0 < top_p_value <= 1.0):
                generation_kwargs["top_p"] = 1.0

        top_k = generation_kwargs.get("top_k")
        if top_k is not None:
            try:
                top_k_value = int(top_k)
            except Exception:
                top_k_value = 0
            if top_k_value < 0:
                generation_kwargs["top_k"] = 0

    try:
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs,
        )
    except RuntimeError as exc:
        message = str(exc)
        is_sampling_prob_error = (
            "probability tensor contains" in message
            or "torch.multinomial" in message
            or "multinomial" in message
        )
        if not is_sampling_prob_error:
            raise

        # Fallback: retry with greedy decoding.
        print(
            "Warning: LLM sampling produced invalid probabilities (NaN/Inf). "
            "Retrying with deterministic decoding (do_sample=False)."
        )
        safe_kwargs = dict(generation_kwargs)
        safe_kwargs["do_sample"] = False
        for key in [
            "temperature",
            "top_p",
            "top_k",
            "typical_p",
            "epsilon_cutoff",
            "eta_cutoff",
        ]:
            safe_kwargs.pop(key, None)
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **safe_kwargs,
        )

    generated_ids = output_ids[0, input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def parse_prediction(response: str):
    """
    Extract predicted class label from generated LLM response.
    
    Args:
        response: Generated response string from generate_response()
    
    Returns:
        str or int: Parsed class label
    """
    if response is None:
        return "Unknown"

    text = response.strip()

    match = re.search(r"predicted\s+class\s+is\s+([0-9]+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"\bclass\b[^0-9]*([0-9]+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"\b([0-9]+)\b", text)
    if match:
        return int(match.group(1))

    match = re.search(r"predicted\s+class\s+is\s+([A-Za-z_]+)", text, re.IGNORECASE)
    if match:
        return match.group(1)

    return "Unknown"



def get_prediction_for_target(model, tokenizer, prompt: str, device: str, **gen_kwargs):
    """
    Convenience wrapper: prompt → generate response → parse prediction.
    
    Args:
        model: Loaded AutoModelForCausalLM model
        tokenizer: Loaded AutoTokenizer
        prompt: Input prompt
        device: Device model is on
        **gen_kwargs: kwargs for model.generate()
    
    Returns:
        str or int: Parsed class label
    """
    response = generate_response(model, tokenizer, prompt, device, **gen_kwargs)
    return parse_prediction(response)


def run_inference_all(model_names: List[str], prompts: List[str], device: str, **gen_kwargs):
    """
    Run inference across multiple LLMs and prompts.
    For each LLM: load model, run all prompts, collect results, then clean up GPU.
    
    Args:
        model_names: List of HuggingFace model names to run
        prompts: List of prompts to send to each LLM
        device: Device to run on ("cuda" or "cpu")
    
    Returns:
        Dict[str, List]: Results organized by model name, e.g.,
                        {"Qwen/Qwen-7B": [pred1, pred2, ...], "meta-llama/Llama-2-7b": [...]}
    """
    print(f"Running inference on device: {device}")

    try:
        from tqdm.auto import tqdm  # type: ignore
    except Exception:
        tqdm = None

    results = {}
    total = int(len(model_names) * len(prompts))
    progress_bar = None
    if tqdm is not None and total > 0:
        progress_bar = tqdm(total=total, desc="LLM inference", unit="prompt")

    completed = 0
    for model_name in model_names:
        tokenizer, model = load_llm(model_name, device)
        predictions = []
        for prompt in prompts:
            pred = get_prediction_for_target(model, tokenizer, prompt, device, **gen_kwargs)
            predictions.append(pred)

            completed += 1
            if progress_bar is not None:
                progress_bar.set_postfix_str(model_name)
                progress_bar.update(1)
            else:
                # Fallback progress indicator (prints ~20 times max).
                if total > 0:
                    step = max(1, total // 20)
                    if completed == 1 or completed % step == 0 or completed == total:
                        pct = 100.0 * completed / total
                        print(f"LLM inference progress: {completed}/{total} ({pct:.1f}%)")
        results[model_name] = predictions
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    if progress_bar is not None:
        progress_bar.close()
    return results

