from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
import re
import json
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
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError(
            "Embedding reduction requires scikit-learn. Install scikit-learn "
            "or leave embedding_max_length unset."
        ) from exc

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


def build_classification_prompt(explanation_text: str, embedding_text: str, subgraph_text: str, template: str):
    """Wrapper for the legacy embedding-classification prompt format."""
    return build_prompt(explanation_text, embedding_text, subgraph_text, template)


def build_raw_reasoning_prompt(raw_features_text: str, neighbor_table_text: str, edge_list_text: str, template: str):
    """Build a prompt for raw graph reasoning without embeddings."""
    return template.format(
        raw_features=raw_features_text,
        neighbor_table=neighbor_table_text,
        edge_list=edge_list_text,
    )


def build_neighbor_selection_prompt(embedding_text: str, candidate_text: str, template: str):
    """Build a prompt for constrained neighbor selection (1-hop reconstruction)."""
    return template.format(embedding=embedding_text, candidates=candidate_text)


def parse_neighbor_selection_response(response: str):
    """Parse a JSON-like neighbor selection response."""
    if response is None:
        return []
    text = response.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "selected_neighbors" in parsed:
            return [int(v) for v in parsed.get("selected_neighbors", [])]
        if isinstance(parsed, list):
            return [int(v) for v in parsed]
    except Exception:
        pass

    # Fallback: extract integers from text
    values = re.findall(r"\b\d+\b", text)
    return [int(v) for v in values]



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
    # For instruct/chat-tuned models (e.g., Qwen-Instruct), wrapping the prompt
    # in the tokenizer's chat template is essential for predictable behavior.
    inputs = None
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            # Compatibility fallback for older/different transformers signatures.
            try:
                formatted_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = tokenizer(formatted_prompt, return_tensors="pt")
            except Exception:
                inputs = None
    if inputs is None:
        inputs = tokenizer(prompt, return_tensors="pt")
    if isinstance(inputs, torch.Tensor):
        input_ids = inputs.to(device)
        attention_mask = None
    else:
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

    text = str(response).strip()
    if not text:
        return "Unknown"

    def _clean_markup(value: str) -> str:
        value = value.strip()
        value = re.sub(r"^```(?:\w+)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
        value = value.strip().strip("`").strip()
        value = re.sub(r"\*+", "", value)
        return value.strip()

    def _label_to_int(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value in (0, 1) else None
        if isinstance(value, float):
            return int(value) if value in (0.0, 1.0) else None

        cleaned = _clean_markup(str(value)).strip().lower()
        cleaned = cleaned.strip(" \t\r\n\"'.,;:()[]{}_*")
        if cleaned in {"0", "0.0"}:
            return 0
        if cleaned in {"1", "1.0"}:
            return 1
        if cleaned in {"licit", "negative", "normal", "benign"}:
            return 0
        if cleaned in {"illicit", "positive", "suspicious", "fraud", "fraudulent"}:
            return 1
        return None

    def _parse_json_candidate(candidate: str) -> Optional[int]:
        cleaned = _clean_markup(candidate)
        try:
            payload = json.loads(cleaned)
        except Exception:
            return None

        if isinstance(payload, dict):
            for key in ("predicted_class", "prediction", "predicted", "label", "class", "answer"):
                if key in payload:
                    parsed = _label_to_int(payload[key])
                    if parsed is not None:
                        return parsed
        elif isinstance(payload, (int, float, str)) and not isinstance(payload, bool):
            return _label_to_int(payload)
        return None

    json_candidates = [text]
    json_candidates.extend(
        match.group(1)
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    )
    json_candidates.extend(
        match.group(0)
        for match in re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL)
    )
    for candidate in json_candidates:
        parsed = _parse_json_candidate(candidate)
        if parsed is not None:
            return parsed

    # Prefer the tail of the response. This keeps prompt echoes/examples from
    # dominating if a model repeats context before giving the final answer.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates = []
    if lines:
        candidates.append(lines[-1])
        candidates.append("\n".join(lines[-2:]))
        candidates.append("\n".join(lines[-5:]))

    marker_matches = list(
        re.finditer(
            r"(?:^|\n)\s*(?:assistant|final\s+answer|answer)\s*[:\n]\s*(.*)$",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if marker_matches:
        candidates.append(marker_matches[-1].group(1))
    candidates.append(text[-1200:])

    def _parse_binary_from(candidate: str) -> Optional[int]:
        cleaned = _clean_markup(candidate)
        direct = _label_to_int(cleaned)
        if direct is not None:
            return direct

        label_token = (
            r"([`*_\"']*(?:[01](?:\.0+)?|licit|illicit|positive|negative|"
            r"normal|benign|suspicious|fraudulent|fraud)[`*_\"']*)"
        )
        relation = r"(?:\s+(?:is|as|would\s+be|should\s+be))?\s*(?:=|:|->|-)?\s*"
        patterns = [
            rf"\b(?:the\s+)?predicted[_\s-]*class\b{relation}{label_token}",
            rf"\b(?:the\s+)?class\b{relation}{label_token}",
            rf"\b(?:the\s+)?(?:answer|prediction|label|classification)\b{relation}{label_token}",
            rf"\b(?:my\s+)?(?:choice|prediction)\b{relation}{label_token}",
            rf"\b(?:i\s+)?(?:choose|pick|select|predict)\s+(?:class\s+)?{label_token}\b",
            rf"\b(?:i\s+)?(?:classify|label)\s+(?:it|this|the\s+transaction|the\s+node)?\s*(?:as\s+)?{label_token}\b",
            rf"\bX\b{relation}{label_token}",
            rf"\b(?:it|transaction|node)\s+(?:is|appears\s+to\s+be|looks)\s+{label_token}",
            rf"(?:^|\n)\s*(?:[-*]\s*)?{label_token}\s*(?:\([^)]*\))?\s*\.?\s*$",
        ]

        for pattern in patterns:
            matches = list(re.finditer(pattern, cleaned, flags=re.IGNORECASE))
            for match in reversed(matches):
                parsed = _label_to_int(match.group(1))
                if parsed is not None:
                    return parsed

        return None

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        key = candidate.strip()
        if key in seen:
            continue
        seen.add(key)
        parsed = _parse_binary_from(candidate)
        if parsed is not None:
            return parsed

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


def run_inference_all(
    model_names: List[str],
    prompts: List[str],
    device: str,
    parse_predictions: bool = True,
    return_raw: bool = False,
    **gen_kwargs,
):
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
        parse_predictions: If True, parse each response as a binary
                           classification label. If False, return raw text.
        return_raw: When parsing predictions, return a dict containing both
                    the raw LLM response and the parsed prediction.
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
            if parse_predictions:
                response = generate_response(model, tokenizer, prompt, device, **gen_kwargs)
                parsed = parse_prediction(response)
                if return_raw:
                    result = {
                        "raw_response": response,
                        "parsed_prediction": parsed,
                    }
                else:
                    result = parsed
            else:
                result = generate_response(model, tokenizer, prompt, device, **gen_kwargs)
            predictions.append(result)

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
