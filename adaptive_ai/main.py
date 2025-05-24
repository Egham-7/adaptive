import litserve as ls  # type: ignore


class AdaptiveModelSelectionAPI(ls.LitAPI):
    def setup(self, _device):
        from services.model_selector import ModelSelector
        from services.prompt_classifier import get_prompt_classifier
        from services.domain_classifier import get_domain_classifier

        self.model_selector = ModelSelector(
            get_prompt_classifier(), get_domain_classifier()
        )

    def decode_request(self, request):
        # Expecting {"prompt": ...}
        return request["prompt"]

    def predict(self, prompts):
        # prompts will be a list if batched, or a single value if not batched
        if isinstance(prompts, list):
            # Batched inference (fastest)
            return [self.model_selector.select_model(p) for p in prompts]
        else:
            # Single inference
            return self.model_selector.select_model(prompts)

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    api = AdaptiveModelSelectionAPI(max_batch_size=8)
    server = ls.LitServer(api, workers_per_device=4)
    server.run(port=8000)
