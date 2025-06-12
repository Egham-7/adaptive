"""
Quantizes the prompt task complexity classifier model using Static Quantization.

This script implements a robust pipeline for custom models:
1. Manually exports the multi-headed PyTorch model to a valid ONNX graph using a wrapper.
2. Downloads the `databricks/databricks-dolly-15k` dataset for calibration.
3. Uses Hugging Face Optimum to perform static quantization on the ONNX model for maximum performance,
   following the latest official API patterns.
4. Saves a complete, portable model artifact including the quantized model, tokenizer, and config.

Requires: pip install torch transformers optimum[onnxruntime] datasets
"""

import argparse
from functools import partial
import logging
from pathlib import Path
import shutil
import sys
import traceback
from typing import cast

from huggingface_hub import PyTorchModelHubMixin
import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- PyTorch Model Definition (from the official model card) ---
class MeanPooling(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class MulticlassHead(nn.Module):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.fc(x))


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        target_sizes: dict[str, int],
        task_type_map: dict[str, str],
        weights_map: dict[str, list[float]],
        divisor_map: dict[str, float],
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
        self.target_names: list[str] = list(target_sizes.keys())
        self.target_sizes_values: list[int] = list(target_sizes.values())
        self.task_type_map: dict[str, str] = task_type_map
        self.weights_map: dict[str, list[float]] = weights_map
        self.divisor_map: dict[str, float] = divisor_map

        self.heads = nn.ModuleList(
            [
                MulticlassHead(self.backbone.config.hidden_size, sz)
                for sz in self.target_sizes_values
            ]
        )
        self.pool = MeanPooling()

    def forward_for_onnx(
        self, input_ids: Tensor, attention_mask: Tensor
    ) -> tuple[Tensor, ...]:
        """A forward pass that returns a tuple of tensors, suitable for ONNX export."""
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.pool(outputs.last_hidden_state, attention_mask)
        return tuple(head(pooled_output) for head in self.heads)


class OnnxExportWrapper(nn.Module):
    """A wrapper to direct the ONNX exporter to the correct forward method."""

    def __init__(self, model: CustomModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, ...]:
        return self.model.forward_for_onnx(input_ids, attention_mask)


# --- Quantization and Inference Logic ---
def quantize_model(
    model_id: str,
    output_dir: Path,
    num_calibration_samples: int,
    seq_length: int = 128,
) -> tuple[Path, Path]:
    """Exports the custom model to ONNX and then quantizes it using Optimum."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import (
        AutoCalibrationConfig,
        AutoQuantizationConfig,
    )

    onnx_dir = output_dir / "onnx_export"
    quantized_dir = output_dir / "quantized_model"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    quantized_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Loading custom PyTorch model and exporting to ONNX...")
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model: CustomModel = CustomModel(
        target_sizes=cast(dict[str, int], config.target_sizes),
        task_type_map=cast(dict[str, str], config.task_type_map),
        weights_map=cast(dict[str, list[float]], config.weights_map),
        divisor_map=cast(dict[str, float], config.divisor_map),
    ).from_pretrained(
        model_id
    )  # type: ignore
    model.eval()

    export_model = OnnxExportWrapper(model)
    onnx_path = onnx_dir / "model.onnx"
    dummy_inputs = tokenizer(
        "This is a test",
        return_tensors="pt",
        max_length=seq_length,
        padding="max_length",
        truncation=True,
    )
    output_names = [f"logits_{i}" for i in range(len(config.target_sizes))]

    torch.onnx.export(
        export_model,
        (dummy_inputs["input_ids"], dummy_inputs["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=output_names,
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            **{name: {0: "batch_size"} for name in output_names},
        },
        opset_version=13,
        do_constant_folding=True,
    )
    logger.info(f"✅ Base ONNX model created at: {onnx_path}")

    logger.info("Step 2: Performing static quantization with Optimum...")
    quantizer = ORTQuantizer.from_pretrained(onnx_dir)

    logger.info("Downloading `databricks/databricks-dolly-15k` for calibration...")

    def preprocess_function(
        examples: dict[str, list[str]], tokenizer: AutoTokenizer
    ) -> dict[str, list[list[int]]]:
        ret_tokenizer: dict[str, list[list[int]]] = tokenizer(
            examples["instruction"],
            padding="max_length",
            truncation=True,
            max_length=seq_length,
        )
        return ret_tokenizer

    calibration_dataset = quantizer.get_calibration_dataset(
        "databricks/databricks-dolly-15k",
        dataset_config_name="default",
        preprocess_function=partial(preprocess_function, tokenizer=tokenizer),
        num_samples=num_calibration_samples,
        dataset_split="train",
    )

    calibration_config = AutoCalibrationConfig.minmax(calibration_dataset)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=True, per_channel=True)

    logger.info("Computing calibration ranges...")
    ranges = quantizer.fit(
        dataset=calibration_dataset,
        calibration_config=calibration_config,
        operators_to_quantize=qconfig.operators_to_quantize,
    )

    logger.info("Saving statically quantized model...")
    try:
        quantizer.quantize(
            save_dir=quantized_dir,
            calibration_tensors_range=ranges,
            quantization_config=qconfig,
        )
    except Exception as e:
        logger.warning(f"AVX512 quantization failed: {e}. Trying ARM64...")
        qconfig = AutoQuantizationConfig.arm64(is_static=True, per_channel=False)
        quantizer.quantize(
            save_dir=quantized_dir,
            calibration_tensors_range=ranges,
            quantization_config=qconfig,
        )

    # The file is saved as model_quantized.onnx inside the save_dir
    quantized_path = quantized_dir / "model_quantized.onnx"
    if not quantized_path.exists():
        raise FileNotFoundError(f"Quantized model not found at {quantized_path}")

    logger.info(f"✅ Statically quantized model created at: {quantized_path}")
    return onnx_path, quantized_path


def main() -> None:
    """Main entry point for the quantization script."""
    parser = argparse.ArgumentParser(
        description="Quantize the prompt task complexity classifier model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_id",
        type=str,
        nargs="?",
        default="nvidia/prompt-task-and-complexity-classifier",
        help="Hugging Face model ID to quantize.",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=5000,
        help="Number of samples to use for static quantization calibration.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for ONNX export.",
    )

    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./quantized_model_output"),
        help="Directory to save the final quantized model and configs.",
    )
    parser.add_argument(
        "--keep_temp",
        action="store_true",
        help="Keep the temporary 'onnx_export' and 'quantized_model' folders.",
    )
    args = parser.parse_args()

    temp_dir = Path("./temp_quantization_artifacts")
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        onnx_path, quantized_path = quantize_model(
            args.model_id,
            temp_dir,
            args.num_calibration_samples,
            args.batch_size,
        )

        logger.info(f"Copying final artifacts to {args.output_dir}...")
        final_onnx_path = args.output_dir / "model.onnx"
        final_quantized_path = args.output_dir / "model_quantized.onnx"

        shutil.copy2(onnx_path, final_onnx_path)
        shutil.copy2(quantized_path, final_quantized_path)

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        tokenizer.save_pretrained(args.output_dir)
        config = AutoConfig.from_pretrained(args.model_id)
        config.save_pretrained(args.output_dir)

        original_size = final_onnx_path.stat().st_size / (1024 * 1024)
        quantized_size = final_quantized_path.stat().st_size / (1024 * 1024)
        logger.info("=" * 50)
        logger.info("🎉 STATIC QUANTIZATION COMPLETE 🎉")
        logger.info(f"Original ONNX model size: {original_size:.2f} MB")
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        logger.info(f"Compression Ratio: {original_size / quantized_size:.1f}x")
        logger.info(f"✅ ONNX model saved to: {final_onnx_path}")
        logger.info(f"✅ Quantized model saved to: {final_quantized_path}")
        logger.info(f"✅ All model artifacts saved to: {args.output_dir}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"❌ An error occurred during the process: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if not args.keep_temp and temp_dir.exists():
            logger.info("Cleaning up temporary artifacts...")
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
