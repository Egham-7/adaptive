from functools import lru_cache
from typing import Any, cast
from pathlib import Path

from huggingface_hub import PyTorchModelHubMixin
from adaptive_ai.core.config import get_settings # Added for settings access

import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import onnxruntime # Added for ONNX inference
from transformers import AutoConfig, AutoModel, AutoTokenizer
# optimum imports are for quantization function, not directly used by PromptClassifier runtime
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import QuantizationConfig, QuantFormat, QuantType


class MeanPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MulticlassHead(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        target_sizes: dict[str, int],
        task_type_map: dict[str, str],
        weights_map: dict[str, list[float]],
        divisor_map: dict[str, float],
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(
            "microsoft/DeBERTa-v3-base", use_safetensors=True
        )
        self.target_sizes = target_sizes.values()
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map
        self.heads = [
            MulticlassHead(self.backbone.config.hidden_size, sz)
            for sz in self.target_sizes
        ]
        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)
        self.pool = MeanPooling()

    def compute_results(
        self, preds: torch.Tensor, target: str, decimal: int = 4
    ) -> tuple[list[str], list[str], list[float]] | list[float]:
        if target == "task_type":
            top2_indices = torch.topk(preds, k=2, dim=1).indices
            softmax_probs = torch.softmax(preds, dim=1)
            top2_probs = softmax_probs.gather(1, top2_indices)
            top2 = top2_indices.detach().cpu().tolist()
            top2_prob = top2_probs.detach().cpu().tolist()
            top2_strings = [
                [self.task_type_map[str(idx)] for idx in sample] for sample in top2
            ]
            top2_prob_rounded = [
                [round(value, 3) for value in sublist] for sublist in top2_prob
            ]
            counter = 0
            for sublist in top2_prob_rounded:
                if sublist[1] < 0.1:
                    top2_strings[counter][1] = "NA"
                counter += 1
            task_type_1 = [sublist[0] for sublist in top2_strings]
            task_type_2 = [sublist[1] for sublist in top2_strings]
            task_type_prob = [sublist[0] for sublist in top2_prob_rounded]
            return (task_type_1, task_type_2, task_type_prob)
        else:
            preds = torch.softmax(preds, dim=1)
            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
            scores = weighted_sum / self.divisor_map[target]
            scores = [round(value, decimal) for value in scores]
            if target == "number_of_few_shots":
                scores = [x if x >= 0.05 else 0 for x in scores]
            return cast(list[float], scores)

    def _extract_classification_results(
        self, logits: list[torch.Tensor]
    ) -> dict[str, list[str] | list[float] | float]:
        """Extract individual classification results from logits."""
        result: dict[str, list[str] | list[float] | float] = {}

        # Task type classification
        task_type_logits = logits[0]
        task_type_results = self.compute_results(task_type_logits, target="task_type")
        if isinstance(task_type_results, tuple):
            result["task_type_1"] = task_type_results[0]
            result["task_type_2"] = task_type_results[1]
            result["task_type_prob"] = task_type_results[2]

        # Other classifications
        classifications = [
            ("creativity_scope", logits[1]),
            ("reasoning", logits[2]),
            ("contextual_knowledge", logits[3]),
            ("number_of_few_shots", logits[4]),
            ("domain_knowledge", logits[5]),
            ("no_label_reason", logits[6]),
            ("constraint_ct", logits[7]),
        ]

        for target, target_logits in classifications:
            target_results = self.compute_results(target_logits, target=target)
            if isinstance(target_results, list):
                result[target] = target_results

        return result

    def _calculate_complexity_scores(
        self,
        results: dict[str, list[str] | list[float] | float],
        task_types: list[str],
    ) -> list[float]:
        """Calculate complexity scores using task-specific weights."""
        # Task type specific weights for complexity calculation
        task_type_weights: dict[str, list[float]] = {
            "Open QA": [0.2, 0.3, 0.15, 0.2, 0.15],
            "Closed QA": [0.1, 0.35, 0.2, 0.25, 0.1],
            "Summarization": [0.2, 0.25, 0.25, 0.1, 0.2],
            "Text Generation": [0.4, 0.2, 0.15, 0.1, 0.15],
            "Code Generation": [0.1, 0.3, 0.2, 0.3, 0.1],
            "Chatbot": [0.25, 0.25, 0.15, 0.1, 0.25],
            "Classification": [0.1, 0.35, 0.25, 0.2, 0.1],
            "Rewrite": [0.2, 0.2, 0.3, 0.1, 0.2],
            "Brainstorming": [0.5, 0.2, 0.1, 0.1, 0.1],
            "Extraction": [0.05, 0.3, 0.3, 0.15, 0.2],
            "Other": [0.25, 0.25, 0.2, 0.15, 0.15],
        }

        # Get required values
        creativity_scope = cast(list[float], results.get("creativity_scope", []))
        reasoning = cast(list[float], results.get("reasoning", []))
        constraint_ct = cast(list[float], results.get("constraint_ct", []))
        domain_knowledge = cast(list[float], results.get("domain_knowledge", []))
        contextual_knowledge = cast(
            list[float], results.get("contextual_knowledge", [])
        )

        complexity_scores = []
        for i, task_type in enumerate(task_types):
            # Use task-specific weights if available, otherwise use default weights
            weights = task_type_weights.get(task_type, [0.3, 0.3, 0.2, 0.1, 0.1])

            score = round(
                weights[0] * creativity_scope[i]
                + weights[1] * reasoning[i]
                + weights[2] * constraint_ct[i]
                + weights[3] * domain_knowledge[i]
                + weights[4] * contextual_knowledge[i],
                5,
            )
            complexity_scores.append(score)

        return complexity_scores

    def _extract_single_sample_results(
        self,
        batch_results: dict[str, list[str] | list[float] | float],
        sample_idx: int,
    ) -> dict[str, list[str] | list[float] | float]:
        """Extract results for a single sample from batch results."""

        single_result: dict[str, list[str] | list[float] | float] = {}

        for key, value in batch_results.items():
            if isinstance(value, list | tuple) and len(value) > sample_idx:
                # Extract the value for this specific sample
                extracted_value = value[sample_idx]
                # Ensure proper typing based on the extracted value
                if isinstance(extracted_value, str):
                    single_result[key] = [extracted_value]  # List[str]
                elif isinstance(extracted_value, int | float):
                    single_result[key] = [float(extracted_value)]  # List[float]
                else:
                    single_result[key] = [extracted_value]
            elif isinstance(value, int | float):
                # Single numeric value
                single_result[key] = float(value)
            else:
                # Handle other cases (should be rare)
                single_result[key] = value

        return single_result

    def process_logits(
        self, logits: list[torch.Tensor]
    ) -> list[dict[str, list[str] | list[float] | float]]:
        """Main orchestration method for processing logits and calculating complexity scores for batched inputs."""
        batch_size = logits[0].shape[0]

        # First, get batch-level results
        batch_results = self._extract_classification_results(logits)

        # Calculate complexity scores for the entire batch
        if "task_type_1" in batch_results:
            task_types = cast(list[str], batch_results["task_type_1"])
            complexity_scores = self._calculate_complexity_scores(
                batch_results, task_types
            )
            batch_results["prompt_complexity_score"] = complexity_scores

        # Now split batch results into individual sample results
        individual_results = []
        for i in range(batch_size):
            single_result = self._extract_single_sample_results(batch_results, i)
            individual_results.append(single_result)

        return individual_results

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        export_mode: bool = False,
    ) -> list[dict[str, list[str] | list[float] | float]] | tuple[torch.Tensor, ...]:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mean_pooled_representation = self.pool(last_hidden_state, attention_mask)
        logits = tuple(
            self.heads[k](mean_pooled_representation)
            for k in range(len(self.target_sizes))
        )
        if export_mode:
            return logits
        return self.process_logits(list(logits))

    def export_to_onnx(self, file_path: str, dummy_input_ids: torch.Tensor, dummy_attention_mask: torch.Tensor) -> None:
        """
        Exports the model to ONNX format.

        Args:
            file_path: The path to save the ONNX model.
            dummy_input_ids: A dummy input tensor for input_ids.
            dummy_attention_mask: A dummy input tensor for attention_mask.
        """
        self.eval()  # Set the model to evaluation mode

        dummy_input = {"input_ids": dummy_input_ids, "attention_mask": dummy_attention_mask}

        input_names = ["input_ids", "attention_mask"]
        output_names = [f"output_{i}" for i in range(len(self.target_sizes))]

        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
        }
        for name in output_names:
            dynamic_axes[name] = {0: "batch_size"}

        # Ensure the model's forward method is called with export_mode=True
        # We need to wrap the model or ensure the arguments are passed correctly
        # to torch.onnx.export

        # Temporarily modify the forward signature for export, if necessary,
        # or use a wrapper. For now, let's assume direct call works with a helper.

        original_forward = self.forward

        # Define a wrapper for the forward method to be used by torch.onnx.export
        def forward_for_export(input_ids_export, attention_mask_export):
            return self.forward({"input_ids": input_ids_export, "attention_mask": attention_mask_export}, export_mode=True)

        # Replace the model's forward method with the wrapper
        self.forward = forward_for_export

        torch.onnx.export(
            self,  # model being run
            (dummy_input_ids, dummy_attention_mask),  # model input (or a tuple for multiple inputs)
            file_path,  # where to save the model (can be a file or file-like object)
            export_params=True,  # store the trained parameter weights inside the model file
            opset_version=11,  # the ONNX version to export the model to
            do_constant_folding=True,  # whether to execute constant folding for optimization
            input_names=input_names,  # the model's input names
            output_names=output_names,  # the model's output names
            dynamic_axes=dynamic_axes,  # variable length axes
        )

        # Restore the original forward method
        self.forward = original_forward

        print(f"Model exported to {file_path}")


class PromptClassifier:
    def __init__(
        self,
        use_quantized_onnx: bool = False,
        onnx_model_path: str | Path | None = None,
    ) -> None:
        self.config = AutoConfig.from_pretrained(
            "nvidia/prompt-task-and-complexity-classifier"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nvidia/prompt-task-and-complexity-classifier"
        )
        # CustomModel instance is always needed for config and processing logic
        self.model = CustomModel(
            target_sizes=self.config.target_sizes,
            task_type_map=self.config.task_type_map,
            weights_map=self.config.weights_map,
            divisor_map=self.config.divisor_map,
        ).from_pretrained("nvidia/prompt-task-and-complexity-classifier")

        self.use_quantized_onnx = use_quantized_onnx
        self.onnx_model_path = onnx_model_path
        self.onnx_session: onnxruntime.InferenceSession | None = None
        self.onnx_input_names: list[str] | None = None
        self.onnx_output_names: list[str] | None = None

        if self.use_quantized_onnx:
            if self.onnx_model_path is None:
                raise ValueError(
                    "onnx_model_path must be provided if use_quantized_onnx is True."
                )
            try:
                self.onnx_session = onnxruntime.InferenceSession(
                    str(self.onnx_model_path), providers=onnxruntime.get_available_providers()
                )
                self.onnx_input_names = [
                    inp.name for inp in self.onnx_session.get_inputs()
                ]
                self.onnx_output_names = [
                    out.name for out in self.onnx_session.get_outputs()
                ]
                print(f"ONNX Runtime session initialized for {self.onnx_model_path}")
                print(f"ONNX Input names: {self.onnx_input_names}")
                print(f"ONNX Output names: {self.onnx_output_names}")
            except Exception as e:
                print(f"Error initializing ONNX session: {e}. Falling back to PyTorch model.")
                self.onnx_session = None

        # Ensure model is in eval mode for PyTorch path
        self.model.eval()


    def classify_prompts(self, prompts: list[str]) -> list[dict[str, Any]]:
        """
        Classify multiple prompts in a batch for optimal GPU utilization.

        Args:
            prompts: List of prompts to classify

        Returns:
            List of classification results, one per prompt
        """
        encoded_texts = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt", # For PyTorch path, ONNX will use .cpu().numpy()
        )

        if self.onnx_session and self.onnx_input_names and self.onnx_output_names:
            print("Classifying prompts using ONNX Runtime.")
            input_ids_np = encoded_texts["input_ids"].cpu().numpy()
            attention_mask_np = encoded_texts["attention_mask"].cpu().numpy()

            # Ensure input names match what the model expects.
            # Common practice is 'input_ids' and 'attention_mask'.
            # If self.onnx_input_names are different, this needs adjustment or ensure export uses these names.
            # Assuming the first input is input_ids and second is attention_mask from export
            input_feed = {
                self.onnx_input_names[0]: input_ids_np,
                self.onnx_input_names[1]: attention_mask_np,
            }

            onnx_outputs_np = self.onnx_session.run(self.onnx_output_names, input_feed)

            # Convert numpy arrays back to torch tensors for process_logits
            # process_logits expects a list of tensors
            onnx_outputs_torch = [torch.from_numpy(arr) for arr in onnx_outputs_np]

            # Use the process_logits method from the CustomModel instance
            # This method handles all the post-processing logic.
            raw_results = self.model.process_logits(onnx_outputs_torch)
        else:
            print("Classifying prompts using PyTorch model.")
            with torch.no_grad():
                raw_results = self.model(encoded_texts)

        # tell MyPy this is indeed list[dict[str,Any]]
        results = cast(list[dict[str, Any]], raw_results)

        print(
            f"Batch classification complete: {len(results)} results for {len(prompts)} prompts (method: {'ONNX' if self.onnx_session else 'PyTorch'})"
        )
        return results

    def classify_task_types(self, texts: list[str]) -> list[str]:
        """
        Extract just the task types from classification results.

        Args:
            texts: List of prompts to classify

        Returns:
            List of primary task types for each prompt
        """
        results = self.classify_prompts(texts)
        task_types = []

        for result in results:
            task_type = result.get("task_type_1", ["Other"])[0]
            task_types.append(task_type)

        return task_types


@lru_cache
def get_prompt_classifier() -> PromptClassifier:
    settings = get_settings()
    return PromptClassifier(
        use_quantized_onnx=settings.model_selection.use_quantized_onnx,
        onnx_model_path=settings.model_selection.onnx_model_path,
    )


def quantize_onnx_model(
    onnx_model_path: str | Path,
    quantized_model_output_path: str | Path,
) -> None:
    """
    Quantizes an ONNX model using Optimum ONNX Runtime.

    Args:
        onnx_model_path: Path to the input ONNX model.
        quantized_model_output_path: Path to save the quantized ONNX model.
    """
    print(f"Loading ONNX model from: {onnx_model_path}")
    onnx_model_path = Path(onnx_model_path)
    quantized_model_output_path = Path(quantized_model_output_path)

    if not onnx_model_path.exists():
        print(f"Error: ONNX model not found at {onnx_model_path}")
        raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

    # Create the quantizer
    quantizer = ORTQuantizer.from_pretrained(onnx_model_path.parent, file_name=onnx_model_path.name)

    # Define the quantization configuration for dynamic quantization
    # Using QInt8 for activations and weights, with QDQ format
    qconfig = QuantizationConfig(
        quant_format=QuantFormat.QDQ,  # QDQ format for quantization
        activation_type=QuantType.QInt8,  # Quantize activations to QInt8
        weight_type=QuantType.QInt8,      # Quantize weights to QInt8
        is_static=False,  # Dynamic quantization
        # For dynamic quantization, per_channel is often not applicable or needed
        # operators_to_quantize=['MatMul', 'Add'] # Optionally specify operators
    )

    print(f"Starting quantization for {onnx_model_path}...")

    # Delete the target file if it exists from a previous run to avoid issues with ORTQuantizer
    if quantized_model_output_path.exists():
        quantized_model_output_path.unlink()

    quantizer.quantize(
        save_dir=quantized_model_output_path.parent,
        quantization_config=qconfig,
        file_name=quantized_model_output_path.name # This should specify the final file name
    )

    if quantized_model_output_path.exists():
        print(f"Successfully quantized model saved to: {quantized_model_output_path}")
    else:
        # Fallback check if the name was auto-generated with _quantized suffix
        # e.g. if onnx_model_path is dir/model.onnx, quantized_model_output_path is dir/qmodel.onnx
        # quantizer might save to dir/model_quantized.onnx
        # This part is tricky with Optimum's API, let's assume file_name works as specified.
        # If not, manual renaming would be needed post-quantization.
        # For now, we trust `file_name` parameter.
        print(f"Warning: Quantized model may not have been saved to the exact path: {quantized_model_output_path}")
        print(f"Please check in {quantized_model_output_path.parent} for a file with a '_quantized' suffix if the expected file is missing.")
