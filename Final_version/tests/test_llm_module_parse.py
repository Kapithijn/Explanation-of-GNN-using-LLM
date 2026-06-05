import unittest

import LLM_Module as llm_module
from Evalueation import compute_classification_metrics
from LLM_Module import generate_response, parse_prediction, run_inference_all


class ParsePredictionTests(unittest.TestCase):
    def test_exact_prompt_format(self):
        self.assertEqual(parse_prediction("The predicted class is 1"), 1)

    def test_markdown_wrapped_binary_label(self):
        self.assertEqual(parse_prediction("The predicted class is: **0** (licit)."), 0)

    def test_json_response(self):
        self.assertEqual(parse_prediction('{"predicted_class": "1"}'), 1)

    def test_fenced_json_response(self):
        response = '```json\n{"answer": 0, "confidence": 0.76}\n```'
        self.assertEqual(parse_prediction(response), 0)

    def test_final_bare_label_after_reasoning(self):
        response = "The embedding resembles the illicit example.\n\nFinal answer:\n1"
        self.assertEqual(parse_prediction(response), 1)

    def test_textual_label_response(self):
        self.assertEqual(parse_prediction("The transaction is illicit."), 1)
        self.assertEqual(parse_prediction("The transaction is suspicious."), 1)

    def test_classification_key_response(self):
        self.assertEqual(parse_prediction("Classification: 0"), 0)

    def test_common_freeform_choice_response(self):
        self.assertEqual(parse_prediction("I choose 0."), 0)
        self.assertEqual(parse_prediction("I predict 1."), 1)
        self.assertEqual(parse_prediction("I would classify it as licit."), 0)

    def test_tail_answer_wins_over_prompt_echo(self):
        response = (
            "Class definition:\n"
            "- 0 = licit\n"
            "- 1 = illicit\n\n"
            "Example\n"
            "Correct label: 1\n\n"
            "Now classify the following case.\n"
            "The predicted class is 0"
        )
        self.assertEqual(parse_prediction(response), 0)

    def test_instruction_without_answer_stays_unknown(self):
        response = "where X is either 0 (licit) or 1 (illicit). Do not output anything else."
        self.assertEqual(parse_prediction(response), "Unknown")


class ClassificationMetricTests(unittest.TestCase):
    def test_unknown_prediction_does_not_crash_metrics(self):
        rows = [{"gnn_pred": 1, "llm_pred": "Unknown"}]
        metrics = compute_classification_metrics(rows)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["parse_rate"], 0.0)
        self.assertEqual(metrics["unknown_n"], 1)
        self.assertEqual(rows[0]["llm_pred"], 2)


class GenerateResponseTests(unittest.TestCase):
    def test_chat_template_tensor_output_can_be_decoded_and_parsed(self):
        class TensorChatTokenizer:
            chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            pad_token_id = 0
            pad_token = "<pad>"
            eos_token = "</s>"

            def apply_chat_template(self, *args, **kwargs):
                return llm_module.torch.tensor([[10, 11]], dtype=llm_module.torch.long)

            def decode(self, generated_ids, skip_special_tokens=True):
                return "The predicted class is 1"

        class FakeModel:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                next_token = llm_module.torch.tensor([[99]], dtype=llm_module.torch.long, device=input_ids.device)
                return llm_module.torch.cat([input_ids, next_token], dim=1)

        response = generate_response(FakeModel(), TensorChatTokenizer(), "prompt", "cpu")
        self.assertEqual(parse_prediction(response), 1)

    def test_run_inference_can_return_raw_text_for_reconstruction(self):
        class TensorChatTokenizer:
            chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            pad_token_id = 0
            pad_token = "<pad>"
            eos_token = "</s>"

            def apply_chat_template(self, *args, **kwargs):
                return llm_module.torch.tensor([[10, 11]], dtype=llm_module.torch.long)

            def decode(self, generated_ids, skip_special_tokens=True):
                return '{"selected_neighbors": [12, 34], "confidence": 0.8}'

        class FakeModel:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                next_token = llm_module.torch.tensor([[99]], dtype=llm_module.torch.long, device=input_ids.device)
                return llm_module.torch.cat([input_ids, next_token], dim=1)

        original_load_llm = llm_module.load_llm
        try:
            llm_module.load_llm = lambda model_name, device: (TensorChatTokenizer(), FakeModel())
            outputs = run_inference_all(["fake-model"], ["prompt"], "cpu", parse_predictions=False)
        finally:
            llm_module.load_llm = original_load_llm

        self.assertEqual(outputs["fake-model"][0], '{"selected_neighbors": [12, 34], "confidence": 0.8}')

    def test_run_inference_can_keep_raw_text_with_parsed_prediction(self):
        class TensorChatTokenizer:
            chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            pad_token_id = 0
            pad_token = "<pad>"
            eos_token = "</s>"

            def apply_chat_template(self, *args, **kwargs):
                return llm_module.torch.tensor([[10, 11]], dtype=llm_module.torch.long)

            def decode(self, generated_ids, skip_special_tokens=True):
                return "I choose 0."

        class FakeModel:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                next_token = llm_module.torch.tensor([[99]], dtype=llm_module.torch.long, device=input_ids.device)
                return llm_module.torch.cat([input_ids, next_token], dim=1)

        original_load_llm = llm_module.load_llm
        try:
            llm_module.load_llm = lambda model_name, device: (TensorChatTokenizer(), FakeModel())
            outputs = run_inference_all(["fake-model"], ["prompt"], "cpu", return_raw=True)
        finally:
            llm_module.load_llm = original_load_llm

        self.assertEqual(outputs["fake-model"][0]["raw_response"], "I choose 0.")
        self.assertEqual(outputs["fake-model"][0]["parsed_prediction"], 0)

    def test_run_inference_supports_batched_prompts(self):
        class BatchTokenizer:
            chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
            pad_token_id = 0
            pad_token = "<pad>"
            eos_token = "</s>"
            padding_side = "right"

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, return_tensors=None):
                if tokenize:
                    return llm_module.torch.tensor([[10, 11]], dtype=llm_module.torch.long)
                return f"chat:{messages[0]['content']}"

            def __call__(self, texts, return_tensors="pt", padding=True):
                batch_size = len(texts)
                input_ids = llm_module.torch.arange(
                    10,
                    10 + batch_size * 3,
                    dtype=llm_module.torch.long,
                ).reshape(batch_size, 3)
                attention_mask = llm_module.torch.ones_like(input_ids)
                return {"input_ids": input_ids, "attention_mask": attention_mask}

            def decode(self, generated_ids, skip_special_tokens=True):
                token = int(generated_ids.reshape(-1)[0].item())
                return f"The predicted class is {token % 2}"

        class FakeModel:
            def generate(self, input_ids, attention_mask=None, **kwargs):
                batch_size = input_ids.shape[0]
                next_tokens = llm_module.torch.arange(
                    101,
                    101 + batch_size,
                    dtype=llm_module.torch.long,
                    device=input_ids.device,
                ).reshape(batch_size, 1)
                return llm_module.torch.cat([input_ids, next_tokens], dim=1)

        original_load_llm = llm_module.load_llm
        try:
            llm_module.load_llm = lambda model_name, device: (BatchTokenizer(), FakeModel())
            outputs = run_inference_all(
                ["fake-model"],
                ["prompt-a", "prompt-b", "prompt-c"],
                "cpu",
                return_raw=True,
                llm_batch_size=2,
            )
        finally:
            llm_module.load_llm = original_load_llm

        self.assertEqual(len(outputs["fake-model"]), 3)
        self.assertEqual(outputs["fake-model"][0]["parsed_prediction"], 1)
        self.assertEqual(outputs["fake-model"][1]["parsed_prediction"], 0)
        self.assertEqual(outputs["fake-model"][2]["parsed_prediction"], 1)


if __name__ == "__main__":
    unittest.main()
