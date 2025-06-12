import pytest
from pathlib import Path
import torch
import numpy as np
import onnxruntime
from unittest.mock import patch, MagicMock

# Assuming adaptive_ai is in PYTHONPATH or installed
from adaptive_ai.services.prompt_classifier import (
    CustomModel,
    PromptClassifier,
    quantize_onnx_model,
    get_prompt_classifier,
)
from adaptive_ai.core.config import Settings, ModelSelectionConfig, AppConfig


# --- Helper Data and Configurations ---

DUMMY_TARGET_SIZES = {"task_type": 3, "creativity_scope": 4}
DUMMY_TASK_TYPE_MAP = {"0": "TaskA", "1": "TaskB", "2": "TaskC"}
DUMMY_WEIGHTS_MAP = {
    "creativity_scope": [0.1, 0.2, 0.3, 0.4],
}
DUMMY_DIVISOR_MAP = {"creativity_scope": 1.0}
DUMMY_MODEL_DIM = 768  # Standard for many base transformers

# --- Pytest Fixtures ---

@pytest.fixture
def dummy_model_args():
    return {
        "target_sizes": DUMMY_TARGET_SIZES,
        "task_type_map": DUMMY_TASK_TYPE_MAP,
        "weights_map": DUMMY_WEIGHTS_MAP,
        "divisor_map": DUMMY_DIVISOR_MAP,
    }

@pytest.fixture
def mock_transformer_model():
    # Mock the backbone model
    mock_model = MagicMock(spec=torch.nn.Module)
    mock_model.config.hidden_size = DUMMY_MODEL_DIM

    # Mock the forward pass output of the backbone
    # It should return an object with a 'last_hidden_state' attribute
    mock_outputs = MagicMock()
    mock_outputs.last_hidden_state = torch.randn(1, 5, DUMMY_MODEL_DIM) # (batch_size, seq_len, hidden_size)
@pytest.fixture
def custom_model_instance(dummy_model_args):
    with patch("transformers.AutoModel.from_pretrained") as mock_backbone_loader:
        # Setup mock for backbone loader (used in CustomModel.__init__)
        mock_model_with_config = MagicMock() # This is the backbone instance
        mock_model_with_config.config = MagicMock() # The backbone instance must have a .config attribute
        mock_model_with_config.config.hidden_size = DUMMY_MODEL_DIM

        # Define the behavior of the backbone when it's called (its forward pass)
        def mock_backbone_forward(input_ids, attention_mask, **kwargs):
            batch_size = input_ids.shape[0]
            seq_len = input_ids.shape[1]
            mock_hidden_state = torch.randn(batch_size, seq_len, DUMMY_MODEL_DIM)
            outputs_obj = MagicMock() # The AutoModel output is an object
            outputs_obj.last_hidden_state = mock_hidden_state
            return outputs_obj

        # Assign this behavior to the mock backbone instance
        # Using .side_effect makes the mock_model_with_config itself callable
        mock_model_with_config.side_effect = mock_backbone_forward

        # Configure the mock_backbone_loader (which is the mock for AutoModel.from_pretrained)
        # to return our specially prepared backbone instance.
        mock_backbone_loader.return_value = mock_model_with_config

        model = CustomModel(**dummy_model_args)
        model.eval() # Set to evaluation mode
        return model


@pytest.fixture
def dummy_inputs():
    input_ids = torch.randint(0, 1000, (1, 5))  # (batch_size, seq_len)
    attention_mask = torch.ones((1, 5))
    return {"input_ids": input_ids, "attention_mask": attention_mask}

@pytest.fixture
def dummy_input_tensors_for_export():
    input_ids = torch.randint(0, 1000, (1, 5), dtype=torch.long)
    attention_mask = torch.ones((1, 5), dtype=torch.long)
    return input_ids, attention_mask


# --- Test Cases ---

def test_custom_model_creation(custom_model_instance):
    assert custom_model_instance is not None
    assert len(custom_model_instance.heads) == len(DUMMY_TARGET_SIZES)

def test_export_custom_model_to_onnx(custom_model_instance, tmp_path, dummy_input_tensors_for_export):
    onnx_model_path = tmp_path / "test_model.onnx"
    input_ids, attention_mask = dummy_input_tensors_for_export

    custom_model_instance.export_to_onnx(str(onnx_model_path), input_ids, attention_mask)

    assert onnx_model_path.exists()
    assert onnx_model_path.stat().st_size > 0

    # Try loading and very basic check
    ort_session = onnxruntime.InferenceSession(str(onnx_model_path))
    assert ort_session is not None

    # Check input/output names based on export logic
    expected_input_names = ["input_ids", "attention_mask"]
    expected_output_names = [f"output_{i}" for i in range(len(DUMMY_TARGET_SIZES))]

    session_input_names = [inp.name for inp in ort_session.get_inputs()]
    session_output_names = [out.name for out in ort_session.get_outputs()]

    assert sorted(session_input_names) == sorted(expected_input_names)
    assert sorted(session_output_names) == sorted(expected_output_names)

    # Basic inference check
    dummy_np_input_ids = input_ids.cpu().numpy()
    dummy_np_attention_mask = attention_mask.cpu().numpy()

    input_feed = {
        "input_ids": dummy_np_input_ids,
        "attention_mask": dummy_np_attention_mask,
    }
    outputs = ort_session.run(None, input_feed)
    assert len(outputs) == len(DUMMY_TARGET_SIZES)


def test_quantize_onnx_model(custom_model_instance, tmp_path, dummy_input_tensors_for_export):
    original_onnx_path = tmp_path / "original.onnx"
    quantized_onnx_path = tmp_path / "quantized_model.onnx"

    input_ids, attention_mask = dummy_input_tensors_for_export
    custom_model_instance.export_to_onnx(str(original_onnx_path), input_ids, attention_mask)
    assert original_onnx_path.exists()

    quantize_onnx_model(original_onnx_path, quantized_onnx_path)

    assert quantized_onnx_path.exists()
    assert quantized_onnx_path.stat().st_size > 0

    # Optional: Check if quantized is smaller (can be flaky, depends on model and quantization effectiveness)
    # assert quantized_onnx_path.stat().st_size < original_onnx_path.stat().st_size

    # Try loading the quantized model
    ort_quantized_session = onnxruntime.InferenceSession(str(quantized_onnx_path))
    assert ort_quantized_session is not None


@patch("adaptive_ai.services.prompt_classifier.get_settings")
@patch("transformers.AutoModel.from_pretrained") # For CustomModel's backbone
@patch.object(CustomModel, "from_pretrained")    # For CustomModel's own weight loading
@patch("transformers.AutoConfig.from_pretrained") # For PromptClassifier's config loading
@patch("transformers.AutoTokenizer.from_pretrained") # For PromptClassifier's tokenizer loading
def test_prompt_classifier_with_onnx(
    mock_auto_tokenizer,
    mock_auto_config,
    mock_custom_model_weights_loader, # Patch for CustomModel.from_pretrained (mixin)
    mock_backbone_loader,             # Patch for AutoModel.from_pretrained (backbone)
    mock_get_settings,
    tmp_path,
    dummy_model_args, # For creating a temporary CustomModel for export
    dummy_input_tensors_for_export
):
    # Configure the mock_backbone_loader. This patch is active for the whole test.
    # It will be used when CustomModel is instantiated, both for the export step
    # and when PromptClassifier instantiates its own CustomModel.
    runtime_mock_backbone_instance = MagicMock()
    runtime_mock_backbone_instance.config = MagicMock()
    runtime_mock_backbone_instance.config.hidden_size = DUMMY_MODEL_DIM
    def runtime_backbone_forward(input_ids, attention_mask, **kwargs):
        batch_size = input_ids.shape[0]; seq_len = input_ids.shape[1]
        outputs = MagicMock(); outputs.last_hidden_state = torch.randn(batch_size, seq_len, DUMMY_MODEL_DIM)
        return outputs
    runtime_mock_backbone_instance.side_effect = runtime_backbone_forward
    mock_backbone_loader.return_value = runtime_mock_backbone_instance # Used by CustomModel in PromptClassifier

    # 1. Prepare ONNX model (export and quantize)
    # For export, create a CustomModel instance. Its backbone will be `runtime_mock_backbone_instance`
    # due to the active `mock_backbone_loader` patch.
    custom_model_for_export = CustomModel(**dummy_model_args)
    custom_model_for_export.eval()

    raw_onnx_path = tmp_path / "cls_raw.onnx"
    quantized_onnx_path = tmp_path / "cls_quantized.onnx"
    input_ids_export, attention_mask_export = dummy_input_tensors_for_export
    custom_model_for_export.export_to_onnx(str(raw_onnx_path), input_ids_export, attention_mask_export)
    quantize_onnx_model(raw_onnx_path, quantized_onnx_path)
    assert quantized_onnx_path.exists()

    # 2. Mock settings for PromptClassifier to use this ONNX model
    mock_settings = Settings(
        app=AppConfig(),
        model_selection=ModelSelectionConfig(
            use_quantized_onnx=True,
            onnx_model_path=str(quantized_onnx_path)
        )
    )
    mock_get_settings.return_value = mock_settings

    # 3. Mocks for PromptClassifier.__init__ dependencies
    # AutoConfig (used by PromptClassifier to get model params for its CustomModel)
    dummy_config_obj = MagicMock()
    dummy_config_obj.target_sizes = DUMMY_TARGET_SIZES
    dummy_config_obj.task_type_map = DUMMY_TASK_TYPE_MAP
    dummy_config_obj.weights_map = DUMMY_WEIGHTS_MAP
    dummy_config_obj.divisor_map = DUMMY_DIVISOR_MAP
    mock_auto_config.return_value = dummy_config_obj

    # AutoTokenizer
    mock_tokenizer_instance = MagicMock()
    def mock_tokenizer_call(texts, **kwargs): # Simulate tokenizer behavior
        return {
            "input_ids": torch.randint(0, 1000, (len(texts), 10), dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 10), dtype=torch.long)
        }
    mock_tokenizer_instance.side_effect = mock_tokenizer_call
    mock_auto_tokenizer.return_value = mock_tokenizer_instance

    # CustomModel.from_pretrained (PyTorchModelHubMixin method)
    # This is called on the CustomModel instance *inside* PromptClassifier.
    # It should return the instance itself, simulating that weights are loaded.
    mock_custom_model_weights_loader.side_effect = lambda instance, *args, **kwargs: instance

    # At this point, when get_prompt_classifier() is called:
    # - It uses mocked settings (directing to use ONNX).
    # - PromptClassifier is initialized:
    #   - Its AutoConfig.from_pretrained call is mocked by mock_auto_config.
    #   - Its AutoTokenizer.from_pretrained call is mocked by mock_auto_tokenizer.
    #   - It creates a CustomModel instance. This CustomModel's __init__ calls
    #     AutoModel.from_pretrained, which is mocked by mock_backbone_loader (returning runtime_mock_backbone_instance).
    #   - The PromptClassifier then calls .from_pretrained() on its CustomModel instance. This call
    #     is mocked by mock_custom_model_weights_loader.
    #   - The ONNX session is loaded because of the mocked settings.

    classifier = get_prompt_classifier()
    assert classifier.onnx_session is not None

    # 4. Classify prompts
    prompts = ["This is a test prompt.", "Another test prompt."]
    results = classifier.classify_prompts(prompts)

    # 5. Verify output structure
    assert isinstance(results, list)
    assert len(results) == len(prompts)
    for res in results:
        assert isinstance(res, dict)
        assert "task_type_1" in res
        assert "creativity_scope" in res

    get_prompt_classifier.cache_clear()

# TODO: Add more tests, e.g. for PromptClassifier without ONNX, edge cases, etc.

# Example of how to run:
# Ensure PYTHONPATH includes the project root if adaptive_ai is not installed.
# pytest adaptive_ai/tests/test_onnx_quantization.py
```
