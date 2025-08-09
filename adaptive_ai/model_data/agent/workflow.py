"""
LangGraph Workflow Definition

Main workflow orchestration for the model enrichment agent.
"""

import logging
import time
from typing import Any

import config
from langgraph.graph import END, StateGraph

from .nodes import ai_analysis_node, extraction_node, search_node, validation_node
from .state import ModelExtractionState
from .utils import (
    ProcessingTracker,
    load_models_to_process,
    update_yaml_file,
)

logger = logging.getLogger(__name__)


def should_retry(state: ModelExtractionState) -> str:
    """Determine if we should retry or finish."""
    if state.get("error_message") and state.get("retry_count", 0) < 2:
        return "search"
    elif (
        state.get("extracted_info")
        and state.get("confidence_score", 0) >= config.MIN_CONFIDENCE_SCORE
    ):
        return "end"
    else:
        return "end"  # Give up after 2 retries


def create_workflow() -> Any:
    """Create and compile the LangGraph workflow."""
    logger.info("🔧 Initializing LangGraph workflow...")

    # Create the graph
    workflow = StateGraph(ModelExtractionState)

    # Add nodes
    workflow.add_node("search", search_node)
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("ai_analysis", ai_analysis_node)
    workflow.add_node("validation", validation_node)

    # Add edges
    workflow.set_entry_point("search")
    workflow.add_edge("search", "extraction")
    workflow.add_edge("extraction", "ai_analysis")
    workflow.add_edge("ai_analysis", "validation")

    # Conditional routing for retries
    workflow.add_conditional_edges(
        "validation",
        should_retry,
        {"search": "search", "end": END},  # Retry from search  # Finish processing
    )

    return workflow.compile()


def process_models_batch(
    models_batch: list[Any],
    app: Any,
    tracker: ProcessingTracker,
) -> None:
    """Process a batch of models using LangGraph workflow"""
    for provider, model_data, yaml_file, model_key in models_batch:
        model_name = model_data["model_name"]

        # Initial state
        initial_state = ModelExtractionState(
            provider=provider,
            model_name=model_name,
            model_data=model_data,
            search_results=[],
            extracted_info=None,
            confidence_score=0.0,
            error_message=None,
            retry_count=0,
        )

        try:
            # Execute LangGraph workflow
            final_state = app.invoke(initial_state)

            # Check if processing was successful
            if (
                final_state.get("extracted_info")
                and final_state.get("confidence_score", 0)
                >= config.MIN_CONFIDENCE_SCORE
            ):

                # Update YAML file
                if update_yaml_file(
                    yaml_file, model_key, final_state["extracted_info"]
                ):
                    tracker.mark_success(provider, model_name)
                else:
                    tracker.mark_failure(provider, model_name)
            else:
                tracker.mark_failure(provider, model_name)

        except Exception as e:
            logger.error(f"Workflow error for {provider}:{model_name}: {e}")
            tracker.mark_failure(provider, model_name)

        # Small delay to be respectful to APIs
        time.sleep(0.1)


def run_enrichment(workflow: Any | None = None) -> None:
    """Main execution function for model enrichment."""
    print("\\n" + "=" * 60)
    print("🚀 LANGGRAPH MODEL ENRICHMENT AGENT")
    print("=" * 60)
    print(
        f"💰 Budget: ${config.TARGET_COST_RANGE[0]}-${config.TARGET_COST_RANGE[1]} "
        f"(Limit: ${config.MAX_COST_LIMIT})"
    )
    print("🧠 AI Model: gpt-4o-mini (cost optimized)")
    print("🔄 LangGraph Workflow with State Management")
    print("🛠️  Multi-source Research + AI Analysis")
    print("=" * 60 + "\\n")

    try:
        # Initialize processing tracker
        tracker = ProcessingTracker()

        # Load models to process
        logger.info("📂 Loading models needing enrichment...")
        models_to_process = load_models_to_process()

        if not models_to_process:
            print("✅ All models are already enriched! Nothing to process.")
            return

        provider_counts: dict[str, int] = {}
        for provider, _, _, _ in models_to_process:
            provider_counts[provider] = provider_counts.get(provider, 0) + 1

        print(f"📦 Found {len(models_to_process)} models needing enrichment:")
        for provider, count in provider_counts.items():
            print(f"   {provider}: {count} models")

        # Estimate costs for target providers only
        estimated_cost = len(models_to_process) * 0.004
        print(
            f"💡 Estimated cost: ${estimated_cost:.2f} "
            f"(Comprehensive enrichment with GPT-4o-mini)"
        )
        print(
            "💡 Target providers: Anthropic, X (Grok), Groq, Google, DeepSeek, OpenAI"
        )

        if estimated_cost > config.MAX_COST_LIMIT:
            print(f"⚠️  Estimated cost exceeds budget limit (${config.MAX_COST_LIMIT})")
            response = input("Continue anyway? (y/N): ")
            if response.lower() != "y":
                print("❌ Aborted by user")
                return

        # Create workflow if not provided
        if workflow is None:
            workflow = create_workflow()

        # Process models in batches
        batch_size = 20  # Process 20 models at a time
        total_batches = (len(models_to_process) + batch_size - 1) // batch_size

        print(
            f"🔄 Processing {len(models_to_process)} models in {total_batches} batches..."
        )

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(models_to_process))
            batch = models_to_process[start_idx:end_idx]

            print(
                f"\\n🔍 Processing batch {batch_num + 1}/{total_batches} "
                f"({len(batch)} models)..."
            )

            # Process batch
            process_models_batch(batch, workflow, tracker)

            # Print progress
            current_processed = end_idx
            tracker.print_progress(
                current_processed, len(models_to_process)
            )

            # Save cache every few batches
            if (batch_num + 1) % 3 == 0:

        # Final summary
        tracker.print_final_summary()


        print("\\n📊 Processing complete! Check YAML files for enriched data.")

        if tracker.failed_count > 0:
            print(
                f"\\n⚠️  {tracker.failed_count} models failed to process. "
                f"You can re-run to retry failed models."
            )

    except KeyboardInterrupt:
        print("\\n⏹️  Processing interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
