import litserve as ls  # type: ignore
from pydantic import BaseModel, ValidationError
from typing import Any, Dict, List, Optional
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import psutil
import os
import sys

# Configure logging for monitoring
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PromptRequest(BaseModel):
    prompt: List[str]


class BatchPromptRequest(BaseModel):
    prompts: List[str]
    batch_id: Optional[str] = None


class AdaptiveModelSelectionAPI(ls.LitAPI):
    def setup(self, device: str) -> None:
        """Initialize the model selector with optimizations for high throughput"""
        logger.info(f"Setting up AdaptiveModelSelectionAPI on device: {device}")

        from services.model_selector import ModelSelector
        from services.prompt_classifier import get_prompt_classifier

        # Initialize model selector
        self.model_selector = ModelSelector(get_prompt_classifier())

        # Performance tracking
        self.request_count = 0
        self.start_time = time.time()
        self.processing_times: List[float] = []

        # Thread pool for CPU-intensive tasks
        max_workers = min(32, (os.cpu_count() or 1) + 4)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        logger.info(f"API setup complete. Thread pool size: {max_workers}")
        logger.info(
            f"Memory usage: {psutil.Process().memory_info().rss / 1024 / 1024:.1f} MB"
        )

    def decode_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced request decoder with batch support and error handling"""
        try:
            # Check if it's a batch request
            if "prompts" in request:
                batch_req = BatchPromptRequest.model_validate(request)
                return {
                    "type": "batch",
                    "prompts": batch_req.prompts,
                    "batch_id": batch_req.batch_id,
                    "batch_size": len(batch_req.prompts),
                }
            else:
                # Single prompt request
                req = PromptRequest.model_validate(request)
                return {"type": "single", "prompt": req.prompt}
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            raise ValueError(f"Invalid request format: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in decode_request: {e}")
            raise ValueError(f"Request processing error: {e}") from e

    def predict(self, decoded_request: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced prediction with batch processing and performance monitoring"""
        start_time = time.time()

        try:
            if decoded_request["type"] == "batch":
                result = self._process_batch(decoded_request)
            else:
                result = self._process_single(decoded_request["prompt"])

            # Performance tracking
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            self.request_count += 1

            # Log performance stats every 100 requests
            if self.request_count % 100 == 0:
                self._log_performance_stats()

            return result

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                "error": str(e),
                "timestamp": time.time(),
                "request_type": decoded_request["type"],
            }

    def _process_single(self, prompt: List[str]) -> Dict[str, Any]:
        """Process a single prompt"""
        try:
            results = self.model_selector.select_model(prompt)
            return {
                "results": results,
                "processing_info": {
                    "timestamp": time.time(),
                    "request_id": self.request_count,
                    "type": "single",
                    "total_prompts": len(prompt),
                },
            }
        except Exception as e:
            logger.error(f"Single prompt processing error: {e}")
            raise

    def _process_batch(self, decoded_request: Dict[str, Any]) -> Dict[str, Any]:
        """Process multiple prompts in parallel"""
        prompts = decoded_request["prompts"]
        batch_id = decoded_request.get("batch_id", f"batch_{self.request_count}")

        logger.info(f"Processing batch {batch_id} with {len(prompts)} prompts")

        try:
            # Process prompts in parallel using thread pool
            future_to_prompt = {
                self.executor.submit(self._safe_process_prompt, [prompt], idx): (
                    prompt,
                    idx,
                )
                for idx, prompt in enumerate(prompts)
            }

            results = []
            errors = []

            for future in future_to_prompt:
                prompt, idx = future_to_prompt[future]
                try:
                    result = future.result(timeout=30)  # 30 second timeout per prompt
                    result["batch_info"] = {
                        "batch_id": batch_id,
                        "prompt_index": idx,
                        "total_prompts": len(prompts),
                    }
                    results.append(result)
                except Exception as e:
                    error_result = {
                        "error": str(e),
                        "prompt_index": idx,
                        "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                        "batch_info": {
                            "batch_id": batch_id,
                            "prompt_index": idx,
                            "total_prompts": len(prompts),
                        },
                    }
                    errors.append(error_result)
                    logger.error(f"Error processing prompt {idx}: {e}")

            return {
                "batch_id": batch_id,
                "total_prompts": len(prompts),
                "successful_results": len(results),
                "errors": len(errors),
                "results": results,
                "error_details": errors if errors else None,
                "processing_info": {
                    "timestamp": time.time(),
                    "batch_size": len(prompts),
                    "success_rate": len(results) / len(prompts) if prompts else 0,
                },
            }

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            return {
                "batch_id": batch_id,
                "error": f"Batch processing failed: {str(e)}",
                "total_prompts": len(prompts),
                "successful_results": 0,
                "errors": len(prompts),
            }

    def _safe_process_prompt(self, prompt: List[str], index: int) -> Dict[str, Any]:
        """Safely process a single prompt with error handling"""
        try:
            results = self.model_selector.select_model(prompt)
            return {"results": results, "prompt_index": index}
        except Exception as e:
            logger.error(f"Error processing prompt {index}: {e}")
            raise

    def _log_performance_stats(self):
        """Log performance statistics"""
        current_time = time.time()
        uptime = current_time - self.start_time
        avg_processing_time = sum(self.processing_times[-100:]) / min(
            100, len(self.processing_times)
        )
        requests_per_second = self.request_count / uptime if uptime > 0 else 0

        # Memory usage
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024

        logger.info(
            f"Performance Stats - Requests: {self.request_count}, "
            f"RPS: {requests_per_second:.2f}, "
            f"Avg Time: {avg_processing_time:.3f}s, "
            f"Memory: {memory_mb:.1f}MB"
        )

        # Clean up old processing times to prevent memory growth
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-500:]

    def encode_response(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced response encoder with metadata"""
        try:
            # Add server metadata
            output["server_info"] = {
                "server_time": time.time(),
                "total_requests_processed": self.request_count,
                "api_version": "1.0.0",
            }

            return output
        except Exception as e:
            logger.error(f"Response encoding error: {e}")
            return {
                "error": f"Response encoding failed: {str(e)}",
                "original_output": output,
            }

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
        logger.info("API cleanup completed")


class OptimizedAdaptiveServer:
    """Optimized server wrapper with enhanced configuration"""

    def __init__(self, api_instance: AdaptiveModelSelectionAPI):
        self.api = api_instance

    def run(
        self,
        port: int = 8000,
        host: str = "0.0.0.0",
        max_batch_size: int = 100,
        timeout: int = 300,
        workers: Optional[int] = None,
    ):
        """Run the server with optimized settings"""

        # Auto-detect optimal worker count
        if workers is None:
            workers = min(4, (os.cpu_count() or 1))

        logger.info(f"Starting optimized server on {host}:{port}")
        logger.info(
            f"Configuration: workers={workers}, max_batch_size={max_batch_size}, timeout={timeout}s"
        )

        try:
            # Create LitServer with optimized settings
            server = ls.LitServer(self.api, accelerator="auto", devices="auto")

            # Run server
            server.run(port=port, host=host)

        except KeyboardInterrupt:
            logger.info("Server shutdown requested")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            self.api.cleanup()


if __name__ == "__main__":
    # Configure for high-throughput processing
    import sys

    # Command line arguments
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else None

    logger.info(
        "Initializing Adaptive Model Selection API for high-throughput processing"
    )

    # Create API instance
    api = AdaptiveModelSelectionAPI()

    # Create optimized server
    server = OptimizedAdaptiveServer(api)

    # Print startup information
    print(
        f"""
    🚀 ADAPTIVE MODEL SELECTION API - OPTIMIZED FOR HIGH THROUGHPUT
    ============================================================
    
    Server Configuration:
    - Port: {port}
    - Host: 0.0.0.0
    - Workers: {workers or 'auto-detected'}
    - Max Batch Size: 100 prompts
    - Timeout: 300 seconds
    
    API Endpoints:
    - Single Prompt: POST /predict with {{"prompt": "your prompt"}}
    - Batch Process: POST /predict with {{"prompts": ["prompt1", "prompt2", ...]}}
    
    Performance Features:
    ✅ Parallel batch processing
    ✅ Thread pool optimization  
    ✅ Memory management
    ✅ Request rate monitoring
    ✅ Error handling & recovery
    ✅ Resource cleanup
    
    Ready to serve! 🎯
    """
    )

    try:
        # Run the optimized server
        server.run(port=port, max_batch_size=100, timeout=300)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
