from functools import lru_cache
import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel, AutoTokenizer


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        )
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class MulticlassHead(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MulticlassHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super(CustomModel, self).__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
        self.target_sizes = target_sizes.values()
        self.task_type_map = task_type_map
        self.weights_map = weights_map
        self.divisor_map = divisor_map
        self.heads = [
            MulticlassHead(self.backbone.config.hidden_size, sz)
            for sz in self.target_sizes
        ]
        for i, head in enumerate(self.heads):
            self.add_module(f"head_{i}", head)
        self.pool = MeanPooling()

    def compute_results(self, preds, target, decimal=4):
        if target == "task_type":
            top2_indices = torch.topk(preds, k=2, dim=1).indices
            softmax_probs = torch.softmax(preds, dim=1)
            top2_probs = softmax_probs.gather(1, top2_indices)
            top2 = top2_indices.detach().cpu().tolist()
            top2_prob = top2_probs.detach().cpu().tolist()
            top2_strings = [
                [self.task_type_map[str(idx)] for idx in sample] for sample in top2
            ]
            top2_prob_rounded = [
                [round(value, 3) for value in sublist] for sublist in top2_prob
            ]
            counter = 0
            for sublist in top2_prob_rounded:
                if sublist[1] < 0.1:
                    top2_strings[counter][1] = "NA"
                counter += 1
            task_type_1 = [sublist[0] for sublist in top2_strings]
            task_type_2 = [sublist[1] for sublist in top2_strings]
            task_type_prob = [sublist[0] for sublist in top2_prob_rounded]
            return (task_type_1, task_type_2, task_type_prob)
        else:
            preds = torch.softmax(preds, dim=1)
            weights = np.array(self.weights_map[target])
            weighted_sum = np.sum(np.array(preds.detach().cpu()) * weights, axis=1)
            scores = weighted_sum / self.divisor_map[target]
            scores = [round(value, decimal) for value in scores]
            if target == "number_of_few_shots":
                scores = [x if x >= 0.05 else 0 for x in scores]
            return scores

    def process_logits(self, logits, domain):
        # Task type specific weights for complexity calculation
        TASK_TYPE_WEIGHTS = {
        "Open QA": [0.2, 0.3, 0.15, 0.2, 0.15],  # Needs reasoning + some domain/contextual recall
        "Closed QA": [0.1, 0.35, 0.2, 0.25, 0.1],  # Factual recall + precise reasoning
        "Summarization": [0.2, 0.25, 0.25, 0.1, 0.2],  # Requires constraint (brevity), context
        "Text Generation": [0.4, 0.2, 0.15, 0.1, 0.15],  # Creativity-driven
        "Code Generation": [0.1, 0.3, 0.2, 0.3, 0.1],  # High constraint & reasoning + domain knowledge
        "Chatbot": [0.25, 0.25, 0.15, 0.1, 0.25],  # Creativity, reasoning, and context
        "Classification": [0.1, 0.35, 0.25, 0.2, 0.1],  # Heavy on reasoning and constraint
        "Rewrite": [0.2, 0.2, 0.3, 0.1, 0.2],  # Needs adherence to form (constraint) + context
        "Brainstorming": [0.5, 0.2, 0.1, 0.1, 0.1],  # Mostly creativity
        "Extraction": [0.05, 0.3, 0.3, 0.15, 0.2],  # Reasoning + strict format (constraint) + some context
        "Other": [0.25, 0.25, 0.2, 0.15, 0.15],  # Balanced default
        }


        result = {}
        # Round 1: "task_type"
        task_type_logits = logits[0]
        task_type_results = self.compute_results(task_type_logits, target="task_type")
        result["task_type_1"] = task_type_results[0]
        result["task_type_2"] = task_type_results[1]
        result["task_type_prob"] = task_type_results[2]

        # Round 2: "creativity_scope"
        creativity_scope_logits = logits[1]
        target = "creativity_scope"
        result[target] = self.compute_results(creativity_scope_logits, target=target)

        # Round 3: "reasoning"
        reasoning_logits = logits[2]
        target = "reasoning"
        result[target] = self.compute_results(reasoning_logits, target=target)

        # Round 4: "contextual_knowledge"
        contextual_knowledge_logits = logits[3]
        target = "contextual_knowledge"
        result[target] = self.compute_results(
            contextual_knowledge_logits, target=target
        )

        # Round 5: "number_of_few_shots"
        number_of_few_shots_logits = logits[4]
        target = "number_of_few_shots"
        result[target] = self.compute_results(number_of_few_shots_logits, target=target)

        # Round 6: "domain_knowledge"
        domain_knowledge_logits = logits[5]
        target = "domain_knowledge"
        result[target] = self.compute_results(domain_knowledge_logits, target=target)

        # Round 7: "no_label_reason"
        no_label_reason_logits = logits[6]
        target = "no_label_reason"
        result[target] = self.compute_results(no_label_reason_logits, target=target)

        # Round 8: "constraint_ct"
        constraint_ct_logits = logits[7]
        target = "constraint_ct"
        result[target] = self.compute_results(constraint_ct_logits, target=target)

        # Get the primary task type
        primary_task_type = result["task_type_1"][0]
        
        # Use task-specific weights if available, otherwise use default weights
        weights = TASK_TYPE_WEIGHTS.get(primary_task_type, [0.3, 0.3, 0.2, 0.1, 0.1])

        # Calculate complexity score using task-specific weights
        result["prompt_complexity_score"] = [
            round(
                weights[0] * creativity
                + weights[1] * reasoning
                + weights[2] * constraint
                + weights[3] * domain_knowledge
                + weights[4] * contextual_knowledge,
                5,
            )
            for creativity, reasoning, constraint, domain_knowledge, contextual_knowledge in zip(
                result["creativity_scope"],
                result["reasoning"],
                result["constraint_ct"],
                result["domain_knowledge"],
                result["contextual_knowledge"],
            )
        ]

        return result

    def forward(self, batch, domain):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        mean_pooled_representation = self.pool(last_hidden_state, attention_mask)
        logits = [
            self.heads[k](mean_pooled_representation)
            for k in range(len(self.target_sizes))
        ]
        return self.process_logits(logits, domain)


config = AutoConfig.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
tokenizer = AutoTokenizer.from_pretrained(
    "nvidia/prompt-task-and-complexity-classifier"
)
model = CustomModel(
    target_sizes=config.target_sizes,
    task_type_map=config.task_type_map,
    weights_map=config.weights_map,
    divisor_map=config.divisor_map,
).from_pretrained("nvidia/prompt-task-and-complexity-classifier")


class PromptClassifier:
    def __init__(self):
        self.model = model
        self.tokenizer = tokenizer

    def classify_prompt(self, prompt, domain):
        encoded_texts = self.tokenizer(
            [prompt],
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            truncation=True,
        )
        result = self.model(encoded_texts, domain)

        return result

    def classify_task_types(self, texts):
        """
        Classify a list of text samples into their respective task types.

        Args:
            texts (list of str): The text samples to classify.

        Returns:
            list of str: The predicted task types for each text sample.
        """
        # Use a default domain since we only care about task type
        default_domain = "Computers_and_Electronics"
        
        results = []
        for text in texts:
            classification = self.classify_prompt(text, default_domain)
            # Get the primary task type (task_type_1)
            results.append(classification["task_type_1"][0])
        
        return results


@lru_cache()
def get_prompt_classifier():
    return PromptClassifier()
