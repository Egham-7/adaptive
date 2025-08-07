"""
LangGraph Workflow Nodes

Individual processing nodes for the model enrichment workflow.
"""

import json
import logging

import config
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .state import ModelExtractionState
from .tools import (
    classify_model_capabilities,
    extract_technical_specs,
    search_model_documentation,
)

logger = logging.getLogger(__name__)


def search_node(state: ModelExtractionState) -> ModelExtractionState:
    """Search for model documentation and specifications."""
    logger.info(f"🔍 Searching for {state['provider']}:{state['model_name']}")

    try:
        search_results = search_model_documentation(
            state["provider"], state["model_name"]
        )

        return {
            **state,
            "search_results": [{"content": search_results, "source": "web_search"}],
            "error_message": None,
        }

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {
            **state,
            "error_message": f"Search failed: {e!s}",
            "retry_count": state.get("retry_count", 0) + 1,
        }


def extraction_node(state: ModelExtractionState) -> ModelExtractionState:
    """Extract technical specifications from search results."""
    logger.info(f"📊 Extracting specs for {state['model_name']}")

    try:
        search_content = "\\n".join(
            [result["content"] for result in state["search_results"]]
        )

        # Extract technical specs
        tech_specs = extract_technical_specs(search_content, state["model_name"])

        # Classify capabilities
        capabilities = classify_model_capabilities(
            state["model_name"], search_content, state["provider"]
        )

        # Combine results
        extracted_info = {
            **tech_specs,
            **capabilities,
            "description": f"AI model {state['model_name']} from {state['provider']}",
            "max_output_tokens": (
                tech_specs["max_context_tokens"] // 4
                if tech_specs.get("max_context_tokens")
                else None
            ),
        }

        return {
            **state,
            "extracted_info": extracted_info,
            "confidence_score": capabilities["confidence_score"],
            "error_message": None,
        }

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {
            **state,
            "error_message": f"Extraction failed: {e!s}",
            "retry_count": state.get("retry_count", 0) + 1,
        }


def ai_analysis_node(state: ModelExtractionState) -> ModelExtractionState:
    """Use GPT-4o-mini for intelligent analysis of search results."""
    logger.info(f"🧠 AI analysis for {state['model_name']}")

    try:
        # Initialize GPT-4o-mini
        from pydantic import SecretStr

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=SecretStr(config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None,
        )

        search_content = "\\n".join(
            [result["content"] for result in state["search_results"]]
        )

        # Create intelligent analysis prompt
        system_prompt = (
            "You are an expert AI model analyst. Analyze the search results "
            "intelligently and make informed decisions about ALL model "
            "characteristics.\\n\\n"
            "IMPORTANT: Base ALL decisions on actual information found in search "
            "results, model names, and provider context. Make intelligent "
            "inferences when direct information isn't available.\\n\\n"
            "For complexity, intelligently assess based on capabilities:\\n"
            '- "easy": Simple, efficient models with basic capabilities '
            "(like mini, lite, small models)\\n"
            '- "medium": Standard models with good general capabilities\\n'
            '- "hard": Advanced models with sophisticated reasoning, '
            "large contexts, complex capabilities"
        )

        task_guidelines = (
            "\\n\\nFor task_type, determine primary strength:\\n"
            '- "Text Generation": General text completion and generation\\n'
            '- "Code Generation": Specialized for coding tasks\\n'
            '- "Chatbot": Optimized for conversational interactions\\n'
            '- "Summarization": Focused on condensing information\\n'
            '- "Classification": Designed for categorization tasks\\n'
            '- "Open QA": Question answering capabilities\\n'
            '- "Other": Specialized models (vision, embedding, etc.)\\n\\n'
            "For model_size_params, extract from name/search results or estimate:\\n"
            '- Look for patterns like "7b", "70b", "175b" in model names\\n'
            '- Use "Unknown" only when no information is available\\n\\n'
            "For latency_tier, intelligently assess:\\n"
            '- "fast": Small/efficient models, quick inference\\n'
            '- "medium": Standard performance models\\n'
            '- "slow": Large models requiring more processing time\\n\\n'
            "Return only valid JSON with NO markdown formatting."
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt + task_guidelines),
                (
                    "user",
                    """Analyze this model comprehensively:

Provider: {provider}
Model Name: {model_name}

Search Results:
{search_content}

Return complete JSON with ALL these fields:
- description: Brief, meaningful description based on actual capabilities
- max_context_tokens: Integer (research-based or intelligent estimate)
- max_output_tokens: Integer (typically 25% of context, but analyze actual limits)
- supports_function_calling: true/false (based on search evidence)
- task_type: Primary capability based on evidence
- complexity: Intelligent assessment of model sophistication
- model_size_params: Extract from name/search or estimate intelligently
- latency_tier: Performance tier based on size and architecture
- languages_supported: Array based on evidence (default ["English"] if unknown)
- confidence_score: Your confidence in this analysis (0.0-1.0)""",
                ),
            ]
        )

        # Format and invoke
        formatted_prompt = prompt.format_messages(
            provider=state["provider"],
            model_name=state["model_name"],
            search_content=search_content[:3000],
        )
        response = llm.invoke(formatted_prompt)

        # Parse JSON response
        try:
            content = (
                str(response.content) if hasattr(response, "content") else str(response)
            )
            if not content:
                raise ValueError("Empty response from LLM")

            # Clean JSON from response
            if content.startswith("```json"):
                json_content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                json_content = content.replace("```", "").strip()
            else:
                json_content = content.strip()

            ai_analysis = json.loads(json_content)

            # Validate response structure
            if not isinstance(ai_analysis, dict):
                raise ValueError(f"Expected dict, got {type(ai_analysis)}")

            # Merge with existing extraction
            existing_info = state.get("extracted_info") or {}
            final_info = {**existing_info, **ai_analysis}

            logger.info(f"AI analysis result keys: {list(ai_analysis.keys())}")
            logger.info(f"Final info keys: {list(final_info.keys())}")

            return {
                **state,
                "extracted_info": final_info,
                "confidence_score": ai_analysis.get("confidence_score", 0.7),
                "error_message": None,
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(
                f"Failed to parse AI response for {state['model_name']}: {e}"
            )
            logger.warning(f"Raw response: {str(response)[:500]}...")
            # Fallback to existing extraction
            return {
                **state,
                "confidence_score": max(0.5, state.get("confidence_score", 0.5)),
                "error_message": None,
            }

    except Exception as e:
        logger.error(f"AI analysis failed: {e}")
        return {
            **state,
            "error_message": f"AI analysis failed: {e!s}",
            "retry_count": state.get("retry_count", 0) + 1,
        }


def validation_node(state: ModelExtractionState) -> ModelExtractionState:
    """Validate extracted information and determine if retry is needed."""
    logger.info(f"✅ Validating {state['model_name']}")

    extracted = state.get("extracted_info") or {}
    confidence = state.get("confidence_score", 0.0)

    # Validation checks
    valid_task_types = [
        "Text Generation",
        "Code Generation",
        "Chatbot",
        "Summarization",
        "Classification",
        "Open QA",
        "Other",
    ]
    valid_complexities = ["easy", "medium", "hard"]

    is_valid = (
        confidence >= config.MIN_CONFIDENCE_SCORE
        and extracted.get("task_type") in valid_task_types
        and extracted.get("complexity") in valid_complexities
        and len(extracted.get("description", "")) > 10
    )

    if is_valid:
        logger.info(f"✅ Validation passed for {state['model_name']}")
        return {**state, "error_message": None}
    else:
        logger.warning(f"⚠️ Validation failed for {state['model_name']}")
        return {
            **state,
            "error_message": "Validation failed: insufficient data quality",
            "retry_count": state.get("retry_count", 0) + 1,
        }
