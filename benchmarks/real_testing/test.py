import os
import pandas as pd
from typing import Dict, List, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv
import time
from multiprocessing import Pool, cpu_count
from functools import partial
from datasets import load_dataset
import logging
from datetime import datetime
import json
from langdetect import detect, LangDetectException

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()
logger.info("Environment variables loaded")

TOKEN_PRICES: Dict[str, Dict[str, float]] = {
    "o3": {"input": 10.00, "output": 40.00},
    "o4-mini": {"input": 1.1, "output": 4.40},
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4.1-mini-2025-04-14": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40},
    "gpt-4.1-2025-04-14": {"input": 2.00, "output": 2.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "deepseek-chat": {"input": 0.07, "output": 1.10},
    "deepseek-reasoner": {"input": 0.14, "output": 2.19},
    "claude-sonnet-4-0": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
    "claude-opus-4-0": {"input": 15.00, "output": 75.00},
}

def is_english_text(text: str) -> bool:
    """Check if the given text is in English"""
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def process_single_row(row: pd.Series, model: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[int]]:
    """Process a single row with API call"""
    logger.info(f"Processing row with model: {model}")
    
    try:
        # Get the conversation list directly
        conversation = row['conversation']
        if not conversation or not isinstance(conversation, list):
            logger.error(f"Invalid conversation format in row: {conversation}")
            return None, None, None, None
            
        # Get the first message which should be user input
        first_message = conversation[0]
        if not isinstance(first_message, dict) or 'content' not in first_message or 'role' not in first_message:
            logger.error(f"Invalid message format in conversation: {first_message}")
            return None, None, None, None
            
        # Verify it's a user message
        if first_message['role'] != 'user':
            logger.error(f"First message is not from user: {first_message['role']}")
            return None, None, None, None
            
        user_input = first_message['content']
        logger.debug(f"Extracted user input: {user_input[:100]}...")
        
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://backend.mangoplant-a7a21605.swedencentral.azurecontainerapps.io/v1"
        )
        
        # Use the entire conversation history for context
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
        
        response = client.chat.completions.create(
            model=model,
            messages=messages
        )
        
        model_name = response.model
        response_content = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        
        logger.info(f"Successfully processed row. Input tokens: {input_tokens}, Output tokens: {output_tokens}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Response Content:\n{response_content}")
        return model_name, response_content, input_tokens, output_tokens
    except Exception as e:
        logger.error(f"Error processing row: {str(e)}", exc_info=True)
        return None, None, None, None

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost based on model and token usage"""
    if model not in TOKEN_PRICES:
        logger.warning(f"Model {model} not found in TOKEN_PRICES")
        return 0.0
    
    prices = TOKEN_PRICES[model]
    input_cost = (input_tokens / 1000000) * prices["input"]  # Convert to thousands of tokens
    output_cost = (output_tokens / 1000000) * prices["output"]  # Convert to thousands of tokens
    return input_cost + output_cost

def process_csv_and_make_api_calls() -> None:
    logger.info("Starting CSV processing and API calls")
    try:
        # Load the dataset from Hugging Face
        logger.info("Loading dataset from Hugging Face")
        ds = load_dataset("lmsys/lmsys-chat-1m")
        # Convert to pandas DataFrame and take first 1000 rows
        df = pd.DataFrame(ds['train'][:1000])
        logger.info(f"Loaded {len(df)} rows from dataset")
        
        # Filter for English prompts
        logger.info("Filtering for English prompts")
        df['is_english'] = df['conversation'].apply(
            lambda x: is_english_text(x[0]['content']) if x and len(x) > 0 and isinstance(x[0], dict) and 'content' in x[0] else False
        )
        df = df[df['is_english']]
        logger.info(f"Found {len(df)} English prompts")
        
        # Add new columns for results
        df['model'] = None
        df['input_tokens'] = None
        df['output_tokens'] = None
        df['response_content'] = None
        df['cost'] = None  # Add cost column
        
        # Number of processes to use (use 75% of available CPUs to avoid overwhelming the system)
        num_processes = max(1, int(cpu_count() * 0.75))
        logger.info(f"Using {num_processes} processes for parallel processing")
        
        # Create a partial function with the model parameter
        model = ""  # You can change this to your preferred model
        process_func = partial(process_single_row, model=model)
        
        # Create a pool of workers
        logger.info("Starting parallel processing")
        with Pool(processes=num_processes) as pool:
            # Process rows in parallel
            results = pool.map(process_func, [row for _, row in df.iterrows()])
        
        # Update DataFrame with results
        success_count = 0
        for i, (model_name, response_content, input_tokens, output_tokens) in enumerate(results):
            if model_name is not None:
                df.at[i, 'model'] = model_name
                df.at[i, 'input_tokens'] = input_tokens
                df.at[i, 'output_tokens'] = output_tokens
                df.at[i, 'response_content'] = response_content
                # Calculate and add cost
                if input_tokens is not None and output_tokens is not None:
                    cost = calculate_cost(model_name, input_tokens, output_tokens)
                    df.at[i, 'cost'] = cost
                else:
                    df.at[i, 'cost'] = 0.0
                success_count += 1
        
        logger.info(f"Successfully processed {success_count} out of {len(df)} rows")
        
        # Save only the required columns to CSV
        output_columns = ['model', 'input_tokens', 'output_tokens', 'response_content', 'cost']
        output_df = df[output_columns]
        
        # Save results to CSV
        output_path = "./LmSys_analysis_results/lmsys_adaptive_result.csv"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        output_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error in process_csv_and_make_api_calls: {str(e)}", exc_info=True)
        raise

def main() -> None:
    logger.info("Starting main execution")
    try:
        process_csv_and_make_api_calls()
        logger.info("Main execution completed successfully")
    except Exception as e:
        logger.error(f"Main execution failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()