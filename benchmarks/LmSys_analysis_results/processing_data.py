import pandas as pd
from datasets import load_dataset
import tiktoken
from typing import Dict, List, Any

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Count tokens using tiktoken library for accurate token counting.
    
    Args:
        text: The text to count tokens for
        model_name: The model name to use for encoding (affects token count)
    """
    if not text:
        return 0
    
    try:
        # Get the appropriate encoding for the model
        if "gpt-4" in model_name.lower():
            encoding = tiktoken.encoding_for_model("gpt-4")
        elif "gpt-3.5" in model_name.lower() or "chatgpt" in model_name.lower():
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        elif "claude" in model_name.lower():
            # Claude uses a similar tokenizer to GPT-3.5, so we'll use that as approximation
            encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        elif "text-davinci" in model_name.lower():
            encoding = tiktoken.encoding_for_model("text-davinci-003")
        else:
            # Default to cl100k_base encoding (used by most modern models)
            encoding = tiktoken.get_encoding("cl100k_base")
        
        # Encode the text and return token count
        tokens = encoding.encode(str(text))
        return len(tokens)
        
    except Exception as e:
        print(f"Error counting tokens for model {model_name}: {e}")
        # Fallback to simple approximation
        tokens = str(text).split()
        return len(tokens)

def extract_lmsys_conversation_data(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract relevant information from LMSYS Chat-1M dataset row.
    """
    result = {
        'model_used': '',
        'input_prompt': '',
        'output_prompt': '',
        'input_tokens': 0,
        'output_tokens': 0
    }
    
    try:
        # Extract model information
        if 'model' in row:
            result['model_used'] = row['model']
        elif 'model_name' in row:
            result['model_used'] = row['model_name']
        
        # Extract conversation data
        # LMSYS dataset typically has 'conversation' field with list of messages
        if 'conversation' in row and row['conversation']:
            conversation = row['conversation']
            
            # Find first user input and first assistant output
            for i, message in enumerate(conversation):
                if isinstance(message, dict):
                    role = message.get('role', '').lower()
                    content = message.get('content', '')
                    
                    # First user message
                    if role in ['user', 'human'] and not result['input_prompt']:
                        result['input_prompt'] = content
                        result['input_tokens'] = count_tokens(content, result['model_used'])
                    
                    # First assistant response (should come after user input)
                    elif role in ['assistant', 'gpt', 'chatgpt', 'claude'] and not result['output_prompt'] and result['input_prompt']:
                        result['output_prompt'] = content
                        result['output_tokens'] = count_tokens(content, result['model_used'])
                        break  # We have both input and output, can stop
        
        # Alternative field names that might exist in LMSYS dataset
        if not result['input_prompt'] and 'prompt' in row:
            result['input_prompt'] = row['prompt']
            result['input_tokens'] = count_tokens(row['prompt'], result['model_used'])
        
        if not result['output_prompt'] and 'response' in row:
            result['output_prompt'] = row['response']
            result['output_tokens'] = count_tokens(row['response'], result['model_used'])
            
    except Exception as e:
        print(f"Error processing row: {e}")
    
    return result

def process_lmsys_dataset(limit: int = None) -> List[Dict[str, Any]]:
    """
    Process LMSYS Chat-1M dataset and extract required information.
    
    Args:
        limit: Optional limit on number of records to process (for testing)
    """
    print("Loading LMSYS Chat-1M dataset...")
    
    try:
        # Load the dataset
        dataset = load_dataset("lmsys/lmsys-chat-1m")
        
        # Get the train split (main data)
        data = dataset['train']
        
        print(f"Dataset loaded. Total records: {len(data)}")
        
        # Limit records if specified (useful for testing)
        if limit:
            data = data.select(range(min(limit, len(data))))
            print(f"Processing limited to {len(data)} records")
        
        results = []
        
        # Process each row
        for i, row in enumerate(data):
            if i % 1000 == 0:  # Progress indicator
                print(f"Processing record {i+1}/{len(data)}")
            
            processed_row = extract_lmsys_conversation_data(row)
            results.append(processed_row)
        
        return results
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def analyze_dataset_structure(sample_size: int = 5):
    """
    Analyze the structure of the LMSYS dataset to understand its format.
    """
    try:
        print("Analyzing dataset structure...")
        dataset = load_dataset("lmsys/lmsys-chat-1m")
        data = dataset['train']
        
        print(f"Dataset info:")
        print(f"- Total records: {len(data)}")
        print(f"- Features: {list(data.features.keys())}")
        
        print(f"\nSample records (first {sample_size}):")
        for i in range(min(sample_size, len(data))):
            print(f"\nRecord {i+1}:")
            row = data[i]
            for key, value in row.items():
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                elif isinstance(value, list) and len(value) > 0:
                    print(f"  {key}: List with {len(value)} items")
                    if isinstance(value[0], dict):
                        print(f"    First item keys: {list(value[0].keys())}")
                else:
                    print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error analyzing dataset: {e}")

def save_results(results: List[Dict[str, Any]], output_file: str = 'lmsys_processed_data.csv'):
    """Save processed results to CSV."""
    if not results:
        print("No data to save.")
        return
    
    df = pd.DataFrame(results)
    
    # Reorder columns to match your requirements
    column_order = ['model_used', 'input_prompt', 'output_prompt', 'input_tokens', 'output_tokens']
    df = df[column_order]
    
    # Remove rows where both input and output are empty
    df = df[(df['input_prompt'] != '') | (df['output_prompt'] != '')]
    
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    print(f"Processed {len(df)} valid records")
    
    # Show some statistics
    print(f"\nDataset Statistics:")
    print(f"- Unique models: {df['model_used'].nunique()}")
    print(f"- Models: {df['model_used'].value_counts().head()}")
    print(f"- Average input tokens: {df['input_tokens'].mean():.2f}")
    print(f"- Average output tokens: {df['output_tokens'].mean():.2f}")

def main():
    """Main processing function."""
    
    # First, analyze the dataset structure
    print("Step 1: Analyzing dataset structure")
    analyze_dataset_structure()
    
    print("\n" + "="*50)
    print("Step 2: Processing dataset")
    
    # Process the dataset (use limit for testing, remove for full processing)
    # For testing, start with a small number like 1000
    results = process_lmsys_dataset()  # Remove limit=1000 for full dataset
    
    if results:
        print("\nStep 3: Displaying sample results")
        # Display first few results for verification
        print("First few processed records:")
        for i, result in enumerate(results[:3]):
            print(f"\nRecord {i+1}:")
            for key, value in result.items():
                if key in ['input_prompt', 'output_prompt'] and len(str(value)) > 100:
                    print(f"  {key}: {str(value)[:100]}...")
                else:
                    print(f"  {key}: {value}")
        
        print("\nStep 4: Saving results")
        save_results(results, 'lmsys_extracted_data.csv')
    else:
        print("No data was processed successfully.")

if __name__ == "__main__":
    # Install required packages if not already installed
    print("Make sure you have the required packages installed:")
    print("pip install datasets pandas tiktoken")
    print("\nNote: tiktoken provides accurate token counting used by OpenAI models")
    print("\n" + "="*50)
    
    main()