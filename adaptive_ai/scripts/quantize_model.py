import argparse
import json
from pathlib import Path
import sys
import torch
import logging # For logging

# Add project root to sys.path for direct script execution
# Assumes the script is in adaptive_ai/scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    from transformers import AutoConfig
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import QuantizationConfig, QuantFormat, QuantType
    # Import necessary components from the adaptive_ai services
    from adaptive_ai.services.prompt_classifier import CustomModel, MeanPooling, MulticlassHead
except ImportError as e:
    # This basic logger will be used if module-level logger setup fails or before it.
    print(f"CRITICAL: Failed to import necessary libraries: {e}. "
          "Please ensure all dependencies (transformers, optimum, onnxruntime) are installed "
          "and that the adaptive_ai package is correctly in PYTHONPATH or installed.", file=sys.stderr)
    sys.exit(1)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomModelExportWrapper(torch.nn.Module):
    """
    A wrapper around CustomModel to provide a forward method suitable for ONNX export,
    which directly returns a tuple of logit tensors.
    """
    def __init__(self, custom_model_instance: CustomModel):
        super().__init__()
        self.model = custom_model_instance
        # Ensure the wrapped model is in evaluation mode
        self.model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Directly use the components of CustomModel to get raw logits
        outputs = self.model.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mean_pooled_representation = self.model.pool(last_hidden_state, attention_mask)
        # Assuming self.model.heads is a list or nn.ModuleList of head modules
        logits = tuple(
            head(mean_pooled_representation) for head in self.model.heads
        )
        return logits


def main():
    parser = argparse.ArgumentParser(
        description="Export a CustomModel to ONNX and quantize it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "model_id_or_path",
        type=str,
        help="Hugging Face model ID or path to local model directory for config loading "
             "(e.g., 'nvidia/prompt-task-and-complexity-classifier').",
    )
    parser.add_argument(
        "--onnx_output_path",
        type=str,
        required=True,
        help="Output path for the exported ONNX model (e.g., models/model.onnx).",
    )
    parser.add_argument(
        "--quantized_output_path",
        type=str,
        required=True,
        help="Output path for the quantized ONNX model (e.g., models/model.quant.onnx).",
    )
    parser.add_argument(
        "--task_type_map_json",
        type=str,
        default=None,
        help="Path to a JSON file for task_type_map (overrides config). Example: '{\"0\": \"TypeA\"}'.",
    )
    parser.add_argument(
        "--weights_map_json",
        type=str,
        default=None,
        help="Path to a JSON file for weights_map (overrides config). Example: '{\"task1\": [0.1, 0.9]}'.",
    )
    parser.add_argument(
        "--divisor_map_json",
        type=str,
        default=None,
        help="Path to a JSON file for divisor_map (overrides config). Example: '{\"task1\": 2.0}'.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for dummy inputs during ONNX export.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=128,
        help="Sequence length for dummy inputs during ONNX export.",
    )

    args = parser.parse_args()

    onnx_output_path = Path(args.onnx_output_path)
    quantized_output_path = Path(args.quantized_output_path)

    try:
        onnx_output_path.parent.mkdir(parents=True, exist_ok=True)
        quantized_output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Error creating output directories: {e}")
        sys.exit(1)

    logger.info(f"Loading AutoConfig from: {args.model_id_or_path}")
    try:
        # This config is used for parameters like target_sizes, maps, vocab_size.
        # CustomModel itself will load its own backbone's config internally.
        hf_config = AutoConfig.from_pretrained(args.model_id_or_path)
    except Exception as e:
        logger.error(f"Failed to load AutoConfig from {args.model_id_or_path}: {e}")
        sys.exit(1)

    # Extract necessary parameters from config or use sensible defaults if not found
    target_sizes = getattr(hf_config, "target_sizes", {"task_type": 8, "creativity_scope": 5}) # Example default
    task_type_map = getattr(hf_config, "task_type_map", {"0": "DefaultType"})
    weights_map = getattr(hf_config, "weights_map", {"default_category": [1.0]})
    divisor_map = getattr(hf_config, "divisor_map", {"default_category": 1.0})
    vocab_size = getattr(hf_config, "vocab_size", 30522) # Default for many BERT-like models

    # Override maps if JSON file paths are provided
    try:
        if args.task_type_map_json:
            logger.info(f"Overriding task_type_map with contents from: {args.task_type_map_json}")
            with open(args.task_type_map_json, 'r') as f:
                task_type_map = json.load(f)
        if args.weights_map_json:
            logger.info(f"Overriding weights_map with contents from: {args.weights_map_json}")
            with open(args.weights_map_json, 'r') as f:
                weights_map = json.load(f)
        if args.divisor_map_json:
            logger.info(f"Overriding divisor_map with contents from: {args.divisor_map_json}")
            with open(args.divisor_map_json, 'r') as f:
                divisor_map = json.load(f)
    except FileNotFoundError as e:
        logger.error(f"Error loading JSON map override: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON map override: {e}")
        sys.exit(1)

    logger.info("Initializing CustomModel...")
    # Note: CustomModel internally loads "microsoft/DeBERTa-v3-base" as its backbone by default.
    try:
        custom_model = CustomModel(
            target_sizes=target_sizes,
            task_type_map=task_type_map,
            weights_map=weights_map,
            divisor_map=divisor_map,
        )
        # The from_pretrained here is from PyTorchModelHubMixin, for loading the whole CustomModel's weights
        # if they were saved that way. For a fresh export based on backbone, this might not be needed,
        # or it should point to where CustomModel's full state (if any) is stored.
        # Assuming we rely on the backbone being loaded by CustomModel's __init__ and want to export that.
        custom_model.eval()
    except Exception as e:
        logger.error(f"Failed to initialize CustomModel: {e}")
        sys.exit(1)

    # --- ONNX Export ---
    logger.info(f"Starting ONNX export to: {args.onnx_output_path}")
    export_model_wrapper = CustomModelExportWrapper(custom_model)

    dummy_input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((args.batch_size, args.seq_length), dtype=torch.long)
    dummy_inputs_tuple = (dummy_input_ids, dummy_attention_mask)

    input_names = ["input_ids", "attention_mask"]
    # Deriving output names based on the number of heads/target_sizes
    output_names = [f"output_{i}" for i in range(len(custom_model.heads))]

    dynamic_axes = {name: {0: "batch_size", 1: "sequence_length"} for name in input_names}
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    try:
        torch.onnx.export(
            export_model_wrapper,
            dummy_inputs_tuple,
            f=str(args.onnx_output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11,
            export_params=True,
            do_constant_folding=True,
        )
        logger.info(f"ONNX model exported successfully to {args.onnx_output_path}")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}", exc_info=True)
        sys.exit(1)

    # --- Quantization ---
    logger.info(f"Starting ONNX quantization. Input: {args.onnx_output_path}, Output: {args.quantized_output_path}")
    try:
        quantizer = ORTQuantizer.from_pretrained(str(onnx_output_path.parent), file_name=onnx_output_path.name)

        quantization_config = QuantizationConfig(
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            is_static=False,
        )

        # Delete target file if it exists to prevent ORTQuantizer error/warning on overwrite
        if quantized_output_path.exists():
            logger.info(f"Pre-existing quantized model found at {quantized_output_path}, deleting it.")
            quantized_output_path.unlink()

        quantizer.quantize(
            save_dir=quantized_output_path.parent,
            file_name=quantized_output_path.name,
            quantization_config=quantization_config,
        )
        logger.info(f"Quantized ONNX model saved successfully to {args.quantized_output_path}")
    except Exception as e:
        logger.error(f"ONNX quantization failed: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Script finished successfully.")


if __name__ == "__main__":
    main()
```
