"""Unit tests for main.py ModelRouterAPI class."""

from unittest.mock import ANY, Mock, patch

import pytest

from adaptive_ai.main import ModelRouterAPI, create_app, setup_logging
from adaptive_ai.models.llm_classification_models import ClassificationResult
from adaptive_ai.models.llm_core_models import (
    ModelCapability,
    ModelSelectionRequest,
    ModelSelectionResponse,
)
from adaptive_ai.models.llm_enums import TaskType

from .models.test_classification_models import create_classification_result

pytestmark = pytest.mark.unit


class TestModelRouterAPI:
    """Test ModelRouterAPI class logic."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings."""
        settings = Mock()
        settings.logging.level = "INFO"
        settings.litserve.max_batch_size = 8
        settings.litserve.batch_timeout = 0.05
        settings.litserve.accelerator = "auto"
        settings.litserve.devices = "auto"
        settings.server.host = "localhost"
        settings.server.port = 8000
        return settings

    @pytest.fixture
    def mock_prompt_classifier(self):
        """Mock prompt classifier."""
        classifier = Mock()
        classifier.classify_prompts.return_value = [
            ClassificationResult(
                task_type_1=["code", "analysis"],
                task_type_2=["generation", "problem_solving"],
                task_type_prob=[0.85],
                creativity_scope=[0.4],
                reasoning=[0.8],
                contextual_knowledge=[0.6],
                prompt_complexity_score=[0.7],
                domain_knowledge=[0.5],
                number_of_few_shots=[0],
                no_label_reason=[0.9],
                constraint_ct=[0.3],
            )
        ]
        return classifier

    @pytest.fixture
    def mock_model_router(self):
        """Mock model router."""
        router = Mock()
        router.select_models.return_value = [
            ModelCapability(provider="openai", model_name="gpt-4"),
            ModelCapability(provider="anthropic", model_name="claude-3-sonnet"),
        ]
        return router

    @pytest.fixture
    def api_instance(self, mock_settings):
        """Create API instance for testing."""
        with (
            patch("adaptive_ai.main.get_settings", return_value=mock_settings),
            patch("adaptive_ai.main.get_prompt_classifier"),
            patch("adaptive_ai.main.ModelRouter"),
        ):
            api = ModelRouterAPI()
            api.setup("cpu")
            return api

    def test_setup(self, mock_settings):
        """Test API setup method."""
        with (
            patch(
                "adaptive_ai.main.get_settings", return_value=mock_settings
            ) as mock_get_settings,
            patch("adaptive_ai.main.get_prompt_classifier") as mock_get_classifier,
            patch("adaptive_ai.main.ModelRouter") as mock_router_class,
        ):

            api = ModelRouterAPI()
            api.setup("cpu")

            # Verify setup was called correctly
            mock_get_settings.assert_called_once()
            mock_get_classifier.assert_called_once_with(lit_logger=api)
            mock_router_class.assert_called_once_with(lit_logger=api)

            assert api.settings == mock_settings
            assert hasattr(api, "prompt_classifier")
            assert hasattr(api, "model_router")

    def test_decode_request(self, api_instance):
        """Test request decoding."""
        request = ModelSelectionRequest(prompt="Test prompt")

        decoded = api_instance.decode_request(request)

        assert decoded == request
        assert isinstance(decoded, ModelSelectionRequest)

    def test_predict_single_request(
        self, api_instance, mock_prompt_classifier, mock_model_router
    ):
        """Test prediction with a single request."""
        api_instance.prompt_classifier = mock_prompt_classifier
        api_instance.model_router = mock_model_router

        requests = [ModelSelectionRequest(prompt="Write Python code", cost_bias=0.5)]

        responses = api_instance.predict(requests)

        assert len(responses) == 1
        response = responses[0]

        assert isinstance(response, ModelSelectionResponse)
        assert response.provider == "openai"
        assert response.model == "gpt-4"
        assert len(response.alternatives) == 1
        assert response.alternatives[0].provider == "anthropic"

    def test_predict_empty_request_list(self, api_instance):
        """Test prediction with empty request list."""
        responses = api_instance.predict([])

        assert len(responses) == 0
        assert isinstance(responses, list)

    def test_predict_multiple_requests(
        self, api_instance, mock_prompt_classifier, mock_model_router
    ):
        """Test prediction with multiple requests."""
        api_instance.prompt_classifier = mock_prompt_classifier
        api_instance.model_router = mock_model_router

        # Mock classifier to return different results for each request
        mock_prompt_classifier.classify_prompts.return_value = [
            ClassificationResult(
                task_type_1=["code"],
                task_type_2=["generation"],
                task_type_prob=[0.8],
                creativity_scope=[0.4],
                reasoning=[0.8],
                contextual_knowledge=[0.6],
                prompt_complexity_score=[0.8],
                domain_knowledge=[0.5],
                number_of_few_shots=[0],
                no_label_reason=[0.9],
                constraint_ct=[0.3],
            ),
            ClassificationResult(
                task_type_1=["chat"],
                task_type_2=["conversation"],
                task_type_prob=[0.7],
                creativity_scope=[0.6],
                reasoning=[0.3],
                contextual_knowledge=[0.4],
                prompt_complexity_score=[0.3],
                domain_knowledge=[0.2],
                number_of_few_shots=[0],
                no_label_reason=[0.8],
                constraint_ct=[0.1],
            ),
        ]

        requests = [
            ModelSelectionRequest(prompt="Write Python code"),
            ModelSelectionRequest(prompt="Hello, how are you?"),
        ]

        responses = api_instance.predict(requests)

        assert len(responses) == 2
        assert all(isinstance(r, ModelSelectionResponse) for r in responses)

    def test_predict_with_error_handling(self, api_instance):
        """Test prediction error handling."""
        # Mock classifier to raise an error
        api_instance.prompt_classifier = Mock()
        api_instance.prompt_classifier.classify_prompts.side_effect = ValueError(
            "Classification failed"
        )

        requests = [ModelSelectionRequest(prompt="Test prompt")]

        with pytest.raises(ValueError, match="Classification failed"):
            api_instance.predict(requests)

    def test_convert_to_task_type(self, api_instance):
        """Test task type conversion."""
        # Test valid task type
        task_type = api_instance._convert_to_task_type(TaskType.CODE_GENERATION.value)
        assert task_type == TaskType.CODE_GENERATION

        # Test invalid task type
        task_type = api_instance._convert_to_task_type("invalid_type")
        assert task_type == TaskType.OTHER

        # Test None input
        task_type = api_instance._convert_to_task_type(None)
        assert task_type == TaskType.OTHER

    def test_process_request_success(self, api_instance, mock_model_router):
        """Test successful request processing."""
        api_instance.model_router = mock_model_router

        request = ModelSelectionRequest(prompt="Test prompt", cost_bias=0.6)
        classification = ClassificationResult(
            task_type_1=["code", "analysis"],
            task_type_2=["generation", "problem_solving"],
            task_type_prob=[0.8],
            creativity_scope=[0.4],
            reasoning=[0.8],
            contextual_knowledge=[0.6],
            prompt_complexity_score=[0.75],
            domain_knowledge=[0.5],
            number_of_few_shots=[0],
            no_label_reason=[0.9],
            constraint_ct=[0.3],
        )

        response = api_instance._process_request(request, classification)

        assert isinstance(response, ModelSelectionResponse)
        assert response.provider == "openai"
        assert response.model == "gpt-4"

        # Verify router was called with correct parameters
        mock_model_router.select_models.assert_called_once()
        call_args = mock_model_router.select_models.call_args
        assert call_args.kwargs["task_complexity"] == 0.75
        assert call_args.kwargs["cost_bias"] == 0.6

    def test_process_request_no_models_found(self, api_instance):
        """Test request processing when no models are found."""
        # Mock router to return empty list
        api_instance.model_router = Mock()
        api_instance.model_router.select_models.return_value = []

        request = ModelSelectionRequest(prompt="Test prompt")
        classification = ClassificationResult(
            task_type_1=["code"],
            task_type_2=["generation"],
            task_type_prob=[0.8],
            creativity_scope=[0.4],
            reasoning=[0.6],
            contextual_knowledge=[0.5],
            prompt_complexity_score=[0.5],
            domain_knowledge=[0.3],
            number_of_few_shots=[0],
            no_label_reason=[0.8],
            constraint_ct=[0.2],
        )

        with pytest.raises(ValueError, match="No eligible models found"):
            api_instance._process_request(request, classification)

    def test_process_request_invalid_model(self, api_instance):
        """Test request processing with invalid model data."""
        # Mock router to return model without required fields
        api_instance.model_router = Mock()
        api_instance.model_router.select_models.return_value = [
            ModelCapability(provider=None, model_name="test-model")  # Missing provider
        ]

        request = ModelSelectionRequest(prompt="Test prompt")
        classification = ClassificationResult(
            task_type_1=["code"],
            task_type_2=["generation"],
            task_type_prob=[0.8],
            creativity_scope=[0.4],
            reasoning=[0.6],
            contextual_knowledge=[0.5],
            prompt_complexity_score=[0.5],
            domain_knowledge=[0.3],
            number_of_few_shots=[0],
            no_label_reason=[0.8],
            constraint_ct=[0.2],
        )

        with pytest.raises(ValueError, match="Selected model missing provider"):
            api_instance._process_request(request, classification)

    def test_classify_prompts_timing(self, api_instance, mock_prompt_classifier):
        """Test that prompt classification includes timing."""
        api_instance.prompt_classifier = mock_prompt_classifier

        # Mock the log method to capture timing
        api_instance.log = Mock()

        result = api_instance._classify_prompts(["Test prompt"])

        assert len(result) == 1

        # Verify timing was logged
        api_instance.log.assert_any_call("task_classification_time", ANY)


class TestCreateApp:
    """Test create_app function."""

    @patch("adaptive_ai.main.get_settings")
    @patch("adaptive_ai.main.ls.LitServer")
    def test_create_app(self, mock_litserver, mock_get_settings):
        """Test app creation."""
        mock_settings = Mock()
        mock_settings.litserve.max_batch_size = 16
        mock_settings.litserve.batch_timeout = 0.02
        mock_settings.litserve.accelerator = "gpu"
        mock_settings.litserve.devices = "0,1"
        mock_get_settings.return_value = mock_settings

        create_app()

        # Verify LitServer was created with correct parameters
        mock_litserver.assert_called_once()
        call_args = mock_litserver.call_args

        # Check that API instance was passed
        api_instance = call_args[0][0]
        assert isinstance(api_instance, ModelRouterAPI)

        # Check keyword arguments
        assert call_args.kwargs["accelerator"] == "gpu"
        assert call_args.kwargs["devices"] == "0,1"
        assert "loggers" in call_args.kwargs


class TestSetupLogging:
    """Test setup_logging function."""

    @patch("adaptive_ai.main.get_settings")
    @patch("adaptive_ai.main.logging")
    def test_setup_logging(self, mock_logging, mock_get_settings):
        """Test logging setup."""
        mock_settings = Mock()
        mock_settings.logging.level = "DEBUG"
        mock_get_settings.return_value = mock_settings

        setup_logging()

        # Verify basic config was called
        mock_logging.basicConfig.assert_called_once()
        call_args = mock_logging.basicConfig.call_args
        assert call_args.kwargs["level"] == mock_logging.DEBUG

        # Verify logger levels were set
        mock_logging.getLogger.assert_called()

    @patch("adaptive_ai.main.get_settings")
    @patch("adaptive_ai.main.logging")
    def test_setup_logging_invalid_level(self, mock_logging, mock_get_settings):
        """Test logging setup with invalid level."""
        mock_settings = Mock()
        mock_settings.logging.level = "INVALID_LEVEL"
        mock_get_settings.return_value = mock_settings

        # Configure mock to not have the invalid attribute, so getattr returns default
        del mock_logging.INVALID_LEVEL

        setup_logging()

        mock_logging.basicConfig.assert_called_once()
        call_args = mock_logging.basicConfig.call_args
        assert call_args.kwargs["level"] == mock_logging.INFO  # Should default to INFO


@pytest.mark.unit
class TestModelRouterAPIEdgeCases:
    """Test edge cases and error scenarios."""

    @patch("adaptive_ai.main.get_settings")
    def test_setup_with_missing_dependencies(self, mock_get_settings):
        """Test setup when dependencies fail to initialize."""
        mock_settings = Mock()
        mock_get_settings.return_value = mock_settings

        with patch(
            "adaptive_ai.main.get_prompt_classifier",
            side_effect=RuntimeError("Classifier failed"),
        ):
            api = ModelRouterAPI()

            # Should handle initialization errors gracefully
            with pytest.raises((RuntimeError, ValueError, ImportError)):
                api.setup("cpu")

    def test_predict_with_malformed_classification(self, test_settings):
        """Test prediction with malformed classification results."""
        with patch("adaptive_ai.main.get_settings", return_value=test_settings):
            api = ModelRouterAPI()
            api.setup("cpu")

            # Mock classifier to return malformed results
            api.prompt_classifier = Mock()
            api.prompt_classifier.classify_prompts.return_value = [
                create_classification_result(task_type_1=[], prompt_complexity_score=[])
            ]

            api.model_router = Mock()
            api.model_router.select_models.return_value = [
                ModelCapability(provider="test", model_name="test-model")
            ]

            requests = [ModelSelectionRequest(prompt="Test")]
            responses = api.predict(requests)

            # Should handle malformed data gracefully
            assert len(responses) == 1
            # Should either succeed with defaults or return error response
            response = responses[0]
            assert isinstance(response, ModelSelectionResponse | dict)
