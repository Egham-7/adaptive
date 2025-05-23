import litserve as ls  # type: ignore


class AdaptiveModelSelectionAPI(ls.LitAPI):
    def setup(self, device):
        from services.model_selector import ModelSelector
        from services.prompt_classifier import get_prompt_classifier
        from services.domain_classifier import get_domain_classifier

        self.model_selector = ModelSelector(
            get_prompt_classifier(), get_domain_classifier()
        )

    def decode_request(self, request):
        # Handles {"prompt": ...} or {"prompt": [...]}
        return request["prompt"]

    def predict_select_model(self, prompt):
        # This method is mapped to POST /select-model
        if isinstance(prompt, list):
            # Batched
            return [self.model_selector.select_model(p) for p in prompt]
        else:
            return self.model_selector.select_model(prompt)

    def encode_response(self, output):
        return output


if __name__ == "__main__":
    server = ls.LitServer(
        AdaptiveModelSelectionAPI(max_batch_size=8),
        accelerator="auto",
        num_workers=2,
    )
    server.run(port=8000)
