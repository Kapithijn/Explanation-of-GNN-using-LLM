from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from Interpertating import get_explanation


def prompt_model_for_explanation(data, prompt, target_node_idx=None, layer="hidden", llm_model="Qwen/Qwen2.5-0.5B-Instruct", tokenizer=None, model=None, device=None, explanations= None):
    """
    This function prompts a language model to generate an explanation for a given graph data input.
    It takes in the graph data, target node index, and layer information, and constructs a prompt to query the language model. The response from the model is returned as an explanation.
    """
    prompt = "You are a helpful assistant that explains the predictions of graph neural networks. Given the following graph data and metadata, provide an explanation for the model's prediction. Try to explain how the model arrived at its decision based on the structure of the graph and the features of the nodes and edges. Focus on the target node and its neighborhood if applicable. Do not retrun a general explanation of the model, but rather an explanation specific to the given graph data and the target node's prediction."
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Select the tokenizer and model, loading them if not provided
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(llm_model)
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            llm_model,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
        )
        model.to(device)

    # Build prompt for the language model with the graph data and any relevant metadata
    meta = []
    if target_node_idx is not None:
        meta.append(f"Target node index: {target_node_idx}")
    if layer:
        meta.append(f"Layer: {layer}")

    prompt_content = prompt + "\n\n"
    if meta:
        prompt_content += " | ".join(meta) + "\n\n"
    prompt_content += str(data)

    messages = [{"role": "user", "content": prompt_content}]

    # Format text for model input
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text = messages[0]["content"]

    inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    output_ids = generated[0][len(inputs.input_ids[0]):]
    response = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"LLM Response: {response}")
    return response


def prompt_model_for_explanation_all_models(model_bundle, data, target_node_idx=None, layer="hidden", explanations_from_explainermodels=None, **kwargs):
    """
    This function generates explanations for all models in the model bundle by prompting a language model.
    It iterates through each model, retrieves the relevant graph data and metadata, and constructs prompts to query the language model for explanations. The responses are collected and returned in a dictionary format.
    """
    explanations = {}
    kwargs.pop('explanations', None)
    for model_name, parts in model_bundle.items():
        explanation = prompt_model_for_explanation(
            data,
            target_node_idx=target_node_idx,
            layer=layer,
            prompt=f"Explain the prediction of {model_name}:",
            explanations=explanations_from_explainermodels.get(model_name) if explanations_from_explainermodels else None,
            **kwargs
        )
        explanations[model_name] = explanation
    return explanations
    