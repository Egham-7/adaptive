#!/usr/bin/env python3
"""
Quantizes the prompt task complexity classifier model.

This script implements a robust pipeline for custom models:
1. Manually exports the multi-headed PyTorch model to a valid ONNX graph using a wrapper.
2. Uses Hugging Face Optimum to quantize the resulting ONNX model.
3. Provides a helper class to demonstrate loading and running the final quantized model.
"""
import argparse
import logging
import shutil
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- PyTorch Model Definition (from the official model card) ---
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super(CustomModel, self).__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
        self.target_names = list(target_sizes.keys())
        self.target_sizes_values = target_sizes.values()
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map

        self.heads = nn.ModuleList(
            [
                MulticlassHead(self.backbone.config.hidden_size, sz)
                for sz in self.target_sizes_values
            ]
        )
        self.pool = MeanPooling()

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Dict[str, List]:
        logits_tuple = self.forward_for_onnx(input_ids, attention_mask)
        return self._process_logits(logits_tuple)

    def forward_for_onnx(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """A forward pass that returns a tuple of tensors, suitable for ONNX export."""
        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooled_output = self.pool(outputs.last_hidden_state, attention_mask)
        return tuple(head(pooled_output) for head in self.heads)

    def _process_logits(
        self, logits_tuple: Tuple[torch.Tensor, ...]
    ) -> Dict[str, List]:
        """Applies post-processing to the raw logits to get the final scores."""
        result = {}
        for i, target in enumerate(self.target_names):
            preds = logits_tuple[i]
            if target == "task_type":
                task_type_results = self._compute_task_type(preds)
                result["task_type_1"] = task_type_results[0]
                result["task_type_2"] = task_type_results[1]
                result["task_type_prob"] = task_type_results[2]
            else:
                result[target] = self._compute_score(preds, target)

        result["prompt_complexity_score"] = [
            round(
                0.35 * c
                + 0.25 * r
                + 0.15 * con
                + 0.15 * dk
                + 0.05 * ck
                + 0.05 * fs,
                5,
            )
            for c, r, con, dk, ck, fs in zip(
                result["creativity_scope"],
                result["reasoning"],
                result["constraint_ct"],
                result["domain_knowledge"],
                result["contextual_knowledge"],
                result["number_of_few_shots"],
            )
        ]
        return result

    def _compute_task_type(self, preds):
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds)
        top2_indices = torch.topk(preds, k=2, dim=1).indices
        softmax_probs = torch.softmax(preds, dim=1)
        top2_probs = softmax_probs.gather(1, top2_indices)
        top2 = top2_indices.detach().cpu().tolist()
        top2_prob = top2_probs.detach().cpu().tolist()
        top2_strings = [
            [self.task_type_map[str(idx)] for idx in sample] for sample in top2
        ]
        top2_prob_rounded = [
            [round(v, 3) for v in sublist] for sublist in top2_prob
        ]
        for i, sublist in enumerate(top2_prob_rounded):
            if sublist[1] < 0.1:
                top2_strings[i][1] = "NA"
        return (
            [s[0] for s in top2_strings],
            [s[1] for s in top2_strings],
            [p[0] for p in top2_prob_rounded],
        )

    def _compute_score(self, preds, target, decimal=4):
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds)
        preds = torch.softmax(preds, dim=1)
        weights = np.array(self.weights_map[target])
        weighted_sum = np.sum(
            np.array(preds.detach().cpu()) * weights, axis=1
        )
        scores = weighted_sum / self.divisor_map[target]
        scores = [round(v, decimal) for v in scores]
        if target == "number_of_few_shots":
            scores = [x if x >= 0.05 else 0 for x in scores]
        return scores


class OnnxExportWrapper(nn.Module):
    """A wrapper to direct the ONNX exporter to the correct forward method."""

    def __init__(self, model: CustomModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model.forward_for_onnx(input_ids, attention_mask)


# --- Quantization and Inference Logic ---
def quantize_model(
    model_id: str, output_dir: Path
) -> Tuple[Path, Path]:
    """Exports the custom model to ONNX and then quantizes it using Optimum."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import (
        AutoQuantizationConfig,
    )

    onnx_dir = output_dir / "onnx_export"
    quantized_dir = output_dir / "quantized_model"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    quantized_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Loading custom PyTorch model using the official method...")
    config = AutoConfig.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = CustomModel(
        target_sizes=config.target_sizes,
        task_type_map=config.task_type_map,
        weights_map=config.weights_map,
        divisor_map=config.divisor_map,
    ).from_pretrained(model_id)
    model.eval()

    logger.info("Step 2: Manually exporting model to ONNX via a wrapper...")
    export_model = OnnxExportWrapper(model)
    onnx_path = onnx_dir / "model.onnx"
    dummy_inputs = tokenizer(
        "This is a test",
        return_tensors="pt",
        max_length=128,
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
    logger.info(f"✅ ONNX model created at: {onnx_path}")

    logger.info("Step 3: Quantizing ONNX model with Optimum...")
    quantizer = ORTQuantizer.from_pretrained(str(onnx_dir))
    qconfig = AutoQuantizationConfig.avx512_vnni(
        is_static=False, per_channel=True
    )

    try:
        quantizer.quantize(save_dir=quantized_dir, quantization_config=qconfig)
    except Exception as e:
        logger.warning(f"AVX512 quantization failed: {e}. Trying ARM64...")
        qconfig = AutoQuantizationConfig.arm64(
            is_static=False, per_channel=False
        )
        quantizer.quantize(save_dir=quantized_dir, quantization_config=qconfig)

    quantized_path = quantized_dir / "model_quantized.onnx"
    if not quantized_path.exists():
        raise FileNotFoundError(
            f"Quantized model not found at {quantized_path}"
        )

    logger.info(f"✅ Quantized model created at: {quantized_path}")
    return onnx_path, quantized_path


def main():
    parser = argparse.ArgumentParser(
        description="Quantize the prompt task complexity classifier model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="nvidia/prompt-task-and-complexity-classifier",
        help="Hugging Face model ID to quantize.",
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
        onnx_path, quantized_path = quantize_model(args.model_id, temp_dir)

        if args.output_dir.exists():
            shutil.rmtree(args.output_dir)
        args.output_dir.mkdir(parents=True)

        logger.info(f"Copying final artifacts to {args.output_dir}...")
        final_model_path = args.output_dir / "model_quantized.onnx"
        shutil.copy2(quantized_path, final_model_path)

        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        tokenizer.save_pretrained(args.output_dir)

        config = AutoConfig.from_pretrained(args.model_id)
        config.save_pretrained(args.output_dir)

        original_size = onnx_path.stat().st_size / (1024 * 1024)
        quantized_size = final_model_path.stat().st_size / (1024 * 1024)
        logger.info("=" * 50)
        logger.info("🎉 QUANTIZATION COMPLETE 🎉")
        logger.info(f"Original ONNX model size: {original_size:.2f} MB")
        logger.info(f"Quantized model size: {quantized_size:.2f} MB")
        logger.info(
            f"Compression Ratio: {original_size / quantized_size:.1f}x"
        )
        logger.info(f"Final model saved to: {args.output_dir}")
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
