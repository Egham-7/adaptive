"""Unit tests for the main ProtocolManagerAPI."""

from unittest.mock import MagicMock, Mock, patch

import pytest  # type: ignore

from adaptive_ai.main import ProtocolManagerAPI
from adaptive_ai.models.llm_classification_models import (
    ClassificationResult,
    DomainClassificationResult,
)
from adaptive_ai.models.llm_core_models import (
    ModelCapability,
    ModelEntry,
    ModelSelectionRequest,
    ProtocolManagerConfig,
)
from adaptive_ai.models.llm_enums import DomainType, ProviderType
from adaptive_ai.models.llm_orchestration_models import (
    OrchestratorResponse,
    ProtocolInfo,
)


class TestProtocolManagerAPI:
    """Test the ProtocolManagerAPI class."""

    @pytest.fixture
    def mock_prompt_classifier(self):
        """Mock prompt classifier."""
        mock = Mock()
        mock.classify_prompts = Mock(
            return_value=[
                ClassificationResult(
                    task_type_1=["Code Generation"],
                    task_type_2=["Other"],
                    task_type_prob=[0.8, 0.2],
                    creativity_scope=[0.5],
                    reasoning=[0.7],
                    contextual_knowledge=[0.6],
                    prompt_complexity_score=[0.6],
                    domain_knowledge=[0.5],
                    number_of_few_shots=[0.0],
                    no_label_reason=[0.1],
                    constraint_ct=[0.3],
                )
            ]
        )
        return mock

    @pytest.fixture
    def mock_domain_classifier(self):
        """Mock domain classifier."""
        mock = Mock()
        mock.classify_domains = Mock(
            return_value=[
                DomainClassificationResult(
                    domain=DomainType.TECHNOLOGY,
                    confidence=0.85,
                    top_3_domains=[DomainType.TECHNOLOGY, DomainType.SCIENCE, DomainType.BUSINESS],
                    top_3_confidences=[0.85, 0.10, 0.05],
                )
            ]
        )
        return mock

    @pytest.fixture
    def mock_model_selector(self):
        """Mock model selection service."""
        mock = Mock()
        mock.select_candidate_models = Mock(
            return_value=[
                ModelEntry(
                    providers=[ProviderType.OPENAI],
                    model_name="gpt-4",
                )
            ]
        )
        mock.enrich_partial_models = Mock(side_effect=lambda x: x)
        return mock

    @pytest.fixture
    def mock_protocol_manager(self):
        """Mock protocol manager."""
        mock = Mock()
        mock.determine_best_protocol = Mock(
            return_value=ProtocolInfo(
                protocol="standard",
                selected_model=ModelEntry(
                    providers=[ProviderType.OPENAI],
                    model_name="gpt-4",
                ),
                openai_params={
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
                has_tools=False,
            )
        )
        return mock

    @pytest.fixture
    def api_instance(
        self,
        mock_prompt_classifier,
        mock_domain_classifier,
        mock_model_selector,
        mock_protocol_manager,
    ):
        """Create API instance with mocked dependencies."""
        with patch("adaptive_ai.main.get_prompt_classifier", return_value=mock_prompt_classifier):
            with patch("adaptive_ai.main.get_domain_classifier", return_value=mock_domain_classifier):
                with patch("adaptive_ai.main.ModelSelectionService", return_value=mock_model_selector):
                    with patch("adaptive_ai.main.ProtocolManager", return_value=mock_protocol_manager):
                        with patch("tiktoken.get_encoding"):
                            api = ProtocolManagerAPI()
                            api.setup("cpu")
                            return api

    def test_setup_initializes_services(self):
        """Test that setup initializes all required services."""
        with patch("adaptive_ai.main.get_prompt_classifier") as mock_get_prompt:
            with patch("adaptive_ai.main.get_domain_classifier") as mock_get_domain:
                with patch("adaptive_ai.main.ModelSelectionService") as mock_model_service:
                    with patch("adaptive_ai.main.ProtocolManager") as mock_protocol:
                        with patch("tiktoken.get_encoding") as mock_tiktoken:
                            api = ProtocolManagerAPI()
                            api.setup("cpu")

                            # Verify all services were initialized
                            mock_get_prompt.assert_called_once()
                            mock_get_domain.assert_called_once()
                            mock_model_service.assert_called_once()
                            mock_protocol.assert_called_once()
                            mock_tiktoken.assert_called_once_with("cl100k_base")

                            assert api.prompt_classifier is not None
                            assert api.domain_classifier is not None
                            assert api.model_selection_service is not None
                            assert api.protocol_manager is not None
                            assert api.tokenizer is not None

    def test_decode_request(self, api_instance):
        """Test request decoding."""
        request_dict = {
            "chat_completion_request": {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gpt-4",
            }
        }

        result = api_instance.decode_request(request_dict)

        assert isinstance(result, ModelSelectionRequest)
        assert result.chat_completion_request["model"] == "gpt-4"
        assert len(result.chat_completion_request["messages"]) == 1

    def test_decode_request_with_protocol_config(self, api_instance):
        """Test request decoding with protocol manager config."""
        request_dict = {
            "chat_completion_request": {
                "messages": [{"role": "user", "content": "Hello"}],
                "model": "gpt-4",
            },
            "protocol_manager_config": {
                "models": [
                    {
                        "provider": "openai",
                        "model_name": "gpt-4",
                        "cost_per_1m_input_tokens": 30.0,
                        "cost_per_1m_output_tokens": 60.0,
                        "max_context_tokens": 8192,
                        "supports_function_calling": True,
                    }
                ],
                "cost_bias": 0.7,
            },
        }

        result = api_instance.decode_request(request_dict)

        assert isinstance(result, ModelSelectionRequest)
        assert result.protocol_manager_config is not None
        assert len(result.protocol_manager_config.models) == 1
        assert result.protocol_manager_config.cost_bias == 0.7

    def test_predict_single_request(self, api_instance):
        """Test prediction with a single request."""
        request = ModelSelectionRequest(
            chat_completion_request={
                "messages": [{"role": "user", "content": "Write a Python function"}],
                "model": "gpt-4",
            }
        )

        results = api_instance.predict([request])

        assert len(results) == 1
        assert isinstance(results[0], OrchestratorResponse)
        assert results[0].protocol_info.protocol == "standard"
        assert results[0].selected_models[0].model_name == "gpt-4"

        # Verify services were called
        api_instance.prompt_classifier.classify_prompts.assert_called_once()
        api_instance.domain_classifier.classify_domains.assert_called_once()
        api_instance.model_selection_service.select_candidate_models.assert_called_once()
        api_instance.protocol_manager.determine_best_protocol.assert_called_once()

    def test_predict_batch_requests(self, api_instance):
        """Test prediction with multiple requests."""
        requests = [
            ModelSelectionRequest(
                chat_completion_request={
                    "messages": [{"role": "user", "content": f"Request {i}"}],
                    "model": "gpt-4",
                }
            )
            for i in range(3)
        ]

        # Mock to return multiple results
        api_instance.prompt_classifier.classify_prompts.return_value = [
            ClassificationResult(
                task_type_1=["Code Generation"],
                task_type_2=["Other"],
                task_type_prob=[0.8, 0.2],
                creativity_scope=[0.5],
                reasoning=[0.7],
                contextual_knowledge=[0.6],
                prompt_complexity_score=[0.6],
                domain_knowledge=[0.5],
                number_of_few_shots=[0.0],
                no_label_reason=[0.1],
                constraint_ct=[0.3],
            )
            for _ in range(3)
        ]

        api_instance.domain_classifier.classify_domains.return_value = [
            DomainClassificationResult(
                domain=DomainType.TECHNOLOGY,
                confidence=0.85,
                top_3_domains=[DomainType.TECHNOLOGY, DomainType.SCIENCE, DomainType.BUSINESS],
                top_3_confidences=[0.85, 0.10, 0.05],
            )
            for _ in range(3)
        ]

        results = api_instance.predict(requests)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, OrchestratorResponse)
            assert result.protocol_info.protocol == "standard"

        # Verify batch processing
        api_instance.prompt_classifier.classify_prompts.assert_called_once_with(
            ["Request 0", "Request 1", "Request 2"]
        )

    def test_predict_with_custom_models(self, api_instance):
        """Test prediction with custom models in protocol config."""
        custom_model = ModelCapability(
            provider="custom-provider",
            model_name="custom-model",
            cost_per_1m_input_tokens=10.0,
            cost_per_1m_output_tokens=20.0,
            max_context_tokens=8192,
            supports_function_calling=False,
        )

        request = ModelSelectionRequest(
            chat_completion_request={
                "messages": [{"role": "user", "content": "Test"}],
                "model": "custom-model",
            },
            protocol_manager_config=ProtocolManagerConfig(
                models=[custom_model],
                cost_bias=0.9,
            ),
        )

        results = api_instance.predict([request])

        assert len(results) == 1
        # Verify model enrichment was called
        api_instance.model_selection_service.enrich_partial_models.assert_called_once()

    def test_predict_with_tools(self, api_instance):
        """Test prediction with tools in the request."""
        # Update mock to indicate tools are present
        api_instance.protocol_manager.determine_best_protocol.return_value = ProtocolInfo(
            protocol="standard",
            selected_model=ModelEntry(
                providers=[ProviderType.OPENAI],
                model_name="gpt-4",
            ),
            openai_params={
                "temperature": 0.7,
                "max_tokens": 2048,
            },
            has_tools=True,
        )

        request = ModelSelectionRequest(
            chat_completion_request={
                "messages": [{"role": "user", "content": "Get weather"}],
                "model": "gpt-4",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather info",
                        },
                    }
                ],
            }
        )

        results = api_instance.predict([request])

        assert len(results) == 1
        assert results[0].protocol_info.has_tools is True

    def test_predict_domain_classification_failure(self, api_instance):
        """Test handling of domain classification failure."""
        # Make domain classification raise an exception
        api_instance.domain_classifier.classify_domains.side_effect = Exception("Classification failed")

        request = ModelSelectionRequest(
            chat_completion_request={
                "messages": [{"role": "user", "content": "Test"}],
                "model": "gpt-4",
            }
        )

        results = api_instance.predict([request])

        # Should still return results with fallback domain
        assert len(results) == 1
        assert results[0].classification_result.domain == DomainType.GENERAL

    def test_predict_empty_message_content(self, api_instance):
        """Test handling of empty message content."""
        request = ModelSelectionRequest(
            chat_completion_request={
                "messages": [{"role": "user", "content": ""}],
                "model": "gpt-4",
            }
        )

        results = api_instance.predict([request])

        assert len(results) == 1
        # Should handle empty content gracefully
        api_instance.prompt_classifier.classify_prompts.assert_called_once_with([""])

    def test_predict_multimodal_content(self, api_instance):
        """Test handling of multimodal message content."""
        request = ModelSelectionRequest(
            chat_completion_request={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {"type": "image_url", "image_url": {"url": "http://example.com/image.jpg"}},
                        ],
                    }
                ],
                "model": "gpt-4-vision",
            }
        )

        results = api_instance.predict([request])

        assert len(results) == 1
        # Should extract text from multimodal content
        api_instance.prompt_classifier.classify_prompts.assert_called_once_with(["What's in this image?"])

    def test_encode_response(self, api_instance):
        """Test response encoding."""
        response = OrchestratorResponse(
            selected_models=[
                ModelEntry(
                    providers=[ProviderType.OPENAI],
                    model_name="gpt-4",
                )
            ],
            protocol_info=ProtocolInfo(
                protocol="standard",
                selected_model=ModelEntry(
                    providers=[ProviderType.OPENAI],
                    model_name="gpt-4",
                ),
                openai_params={
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
                has_tools=False,
            ),
            classification_result=ClassificationResult(
                task_type_1=["Code Generation"],
                task_type_2=["Other"],
                task_type_prob=[0.8, 0.2],
                creativity_scope=[0.5],
                reasoning=[0.7],
                contextual_knowledge=[0.6],
                prompt_complexity_score=[0.6],
                domain_knowledge=[0.5],
                number_of_few_shots=[0.0],
                no_label_reason=[0.1],
                constraint_ct=[0.3],
                domain=DomainType.TECHNOLOGY,
            ),
        )

        encoded = api_instance.encode_response([response])

        assert "output" in encoded
        assert len(encoded["output"]) == 1
        assert "selected_models" in encoded["output"][0]
        assert "protocol_info" in encoded["output"][0]
        assert "classification_result" in encoded["output"][0]

    def test_token_counting(self, api_instance):
        """Test token counting functionality."""
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.encode = Mock(return_value=list(range(100)))  # 100 tokens
        api_instance.tokenizer = mock_tokenizer

        request = ModelSelectionRequest(
            chat_completion_request={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Write code"},
                ],
                "model": "gpt-4",
            }
        )

        results = api_instance.predict([request])

        # Verify tokenizer was used
        assert mock_tokenizer.encode.call_count >= 1

    def test_logging_functionality(self, api_instance):
        """Test that logging is called during prediction."""
        with patch.object(api_instance, "log") as mock_log:
            request = ModelSelectionRequest(
                chat_completion_request={
                    "messages": [{"role": "user", "content": "Test"}],
                    "model": "gpt-4",
                }
            )

            api_instance.predict([request])

            # Verify logging was called
            assert mock_log.call_count > 0
            # Should log classification times
            mock_log.assert_any_call("task_classification_time", mock.ANY)
            mock_log.assert_any_call("domain_classification_time", mock.ANY)

    def test_batch_validation(self, api_instance):
        """Test that batch sizes are validated correctly."""
        # Create mismatched mock returns (wrong batch size)
        api_instance.prompt_classifier.classify_prompts.return_value = [
            ClassificationResult(
                task_type_1=["Code Generation"],
                task_type_2=["Other"],
                task_type_prob=[0.8, 0.2],
                creativity_scope=[0.5],
                reasoning=[0.7],
                contextual_knowledge=[0.6],
                prompt_complexity_score=[0.6],
                domain_knowledge=[0.5],
                number_of_few_shots=[0.0],
                no_label_reason=[0.1],
                constraint_ct=[0.3],
            )
        ]  # Only 1 result

        requests = [
            ModelSelectionRequest(
                chat_completion_request={
                    "messages": [{"role": "user", "content": f"Request {i}"}],
                    "model": "gpt-4",
                }
            )
            for i in range(3)  # 3 requests
        ]

        # Should handle mismatch gracefully
        results = api_instance.predict(requests)
        assert len(results) == 3  # Should still return 3 results

    def test_model_enrichment_with_partial_models(self, api_instance):
        """Test model enrichment for partial model specifications."""
        partial_model = ModelCapability(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            # Missing cost and context info
        )

        enriched_model = ModelCapability(
            provider=ProviderType.OPENAI,
            model_name="gpt-4",
            cost_per_1m_input_tokens=30.0,
            cost_per_1m_output_tokens=60.0,
            max_context_tokens=8192,
            supports_function_calling=True,
        )

        api_instance.model_selection_service.enrich_partial_models.return_value = [enriched_model]

        request = ModelSelectionRequest(
            chat_completion_request={
                "messages": [{"role": "user", "content": "Test"}],
                "model": "gpt-4",
            },
            protocol_manager_config=ProtocolManagerConfig(
                models=[partial_model],
            ),
        )

        results = api_instance.predict([request])

        assert len(results) == 1
        # Verify enrichment was called
        api_instance.model_selection_service.enrich_partial_models.assert_called_once_with([partial_model])


class TestLitServeIntegration:
    """Test LitServe-specific functionality."""

    def test_create_app(self):
        """Test app creation."""
        from adaptive_ai.main import create_app

        with patch("adaptive_ai.main.ls.LitServer") as mock_server:
            app = create_app()

            # Verify LitServer was created with correct parameters
            mock_server.assert_called_once()
            call_args = mock_server.call_args
            assert call_args[1]["accelerator"] == "auto"
            assert call_args[1]["max_batch_size"] == 32

    def test_console_logger(self):
        """Test custom console logger."""
        from adaptive_ai.main import ConsoleLogger

        logger = ConsoleLogger()

        with patch("builtins.print") as mock_print:
            logger.process("test_key", "test_value")

            mock_print.assert_called_once_with(
                "[LitServe] Received test_key with value test_value", 
                flush=True
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])