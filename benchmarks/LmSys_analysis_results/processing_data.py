import json
from typing import Any

import pandas as pd
import tiktoken
from datasets import load_dataset


def count_tokens(text: str | Any, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens using tiktoken library for accurate token counting.
    """
    try:
        # Initialize tokenizer
        enc = tiktoken.encoding_for_model(model_name)
        # Convert text to string if it's not already
        text_str = str(text)
        # Count tokens
        return len(enc.encode(text_str))
    except Exception as e:
        print(f"Error counting tokens: {e}")
        # Fallback to simple word count
        return len(str(text).split())


def extract_lmsys_conversation_data(row: dict[str, Any]) -> dict[str, Any]:
    """
    Extract relevant information from LMSYS Chat-1M dataset row.
    """
    try:
        # Extract basic information
        conversation = row.get("conversation", [])
        model_used = str(row.get("model", "unknown"))
        language = str(row.get("language", "unknown"))
        turn = int(row.get("turn", 0))

        # Initialize result dictionary
        result: dict[str, Any] = {
            "model_used": model_used,
            "language": language,
            "turn": turn,
            "conversation": [],
            "first_user_input": "",
            "first_assistant_output": "",
            "total_tokens": 0,
        }

        # Process conversation
        if conversation:
            # Find first user input and first assistant output
            for _i, message in enumerate(conversation):
                if isinstance(message, dict):
                    role = message.get("role", "").lower()
                    content = message.get("content", "")

                    # Add to conversation list
                    result["conversation"].append({"role": role, "content": content})

                    # Track first user input and assistant output
                    if role == "user" and not result["first_user_input"]:
                        result["first_user_input"] = content
                    elif role == "assistant" and not result["first_assistant_output"]:
                        result["first_assistant_output"] = content

                    # Count tokens
                    result["total_tokens"] += count_tokens(content, model_used)

        return result

    except Exception as e:
        print(f"Error processing row: {e}")
        return {
            "model_used": "error",
            "language": "error",
            "turn": 0,
            "conversation": [],
            "first_user_input": "",
            "first_assistant_output": "",
            "total_tokens": 0,
        }


def process_lmsys_dataset(limit: int | None = None) -> list[dict[str, Any]]:
    """
    Process LMSYS Chat-1M dataset and extract required information.
    """
    try:
        # Load dataset
        dataset = load_dataset("lmsys/lmsys-chat-1m")
        data = dataset["train"]

        # Apply limit if specified
        if limit is not None:
            data = data.select(range(min(limit, len(data))))
            print(f"Processing limited to {len(data)} records")

        results: list[dict[str, Any]] = []

        # Process each row
        for row in data:
            processed_row = extract_lmsys_conversation_data(row)
            results.append(processed_row)

        return results

    except Exception as e:
        print(f"Error processing dataset: {e}")
        return []


def analyze_dataset_structure(sample_size: int = 5):
    """
    Analyze and print the structure of the LMSYS Chat-1M dataset.
    """
    try:
        # Load dataset
        dataset = load_dataset("lmsys/lmsys-chat-1m")
        data = dataset["train"]

        print("Dataset info:")
        print(f"- Total records: {len(data)}")
        print(f"- Features: {list(data.features.keys())}")

        # Print sample records
        print("\nSample records:")
        for i, row in enumerate(data.select(range(sample_size))):
            print(f"\nRecord {i + 1}:")
            print(json.dumps(row, indent=2))

    except Exception as e:
        print(f"Error analyzing dataset: {e}")


def save_results(
    results: list[dict[str, Any]], output_file: str = "lmsys_processed_data.csv"
):
    """Save processed results to CSV."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

        # Show some statistics
        print("\nDataset Statistics:")
        print(f"- Unique models: {df['model_used'].nunique()}")
        print(f"- Models: {df['model_used'].value_counts().head()}")

    except Exception as e:
        print(f"Error saving results: {e}")


def main():
    # Analyze dataset structure
    analyze_dataset_structure()

    # Process dataset
    results = process_lmsys_dataset(limit=1000)  # Process first 1000 records

    # Save results
    save_results(results)


if __name__ == "__main__":
    # Install required packages if not already installed
    print("Make sure you have the required packages installed:")
    print("pip install datasets pandas tiktoken")
    print("\nNote: tiktoken provides accurate token counting used by OpenAI models")
    print("\n" + "=" * 50)

    main()
