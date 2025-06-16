import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import numpy as np

# Token pricing per million tokens
TOKEN_PRICES: Dict[str, Dict[str, float]] = {
    "o3": {"input": 10.00, "output": 40.00},
    "o4-mini": {"input": 1.1, "output": 4.40},
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano-2025-04-14": {"input": 0.10, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "output": 2.00},
    "gpt-4.1-2025-04-14": {"input": 1.50, "output": 1.50},
    "gpt-4.1-mini-2025-04-14": {"input": 0.35, "output": 1.40},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "deepseek-chat": {"input": 0.07, "output": 1.10},
    "deepseek-reasoner": {"input": 0.14, "output": 2.19},
    "claude-sonnet-4-0": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
    "claude-opus-4-0": {"input": 15.00, "output": 75.00},
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost based on model and token usage"""
    # Handle model name variations
    if model == "gpt-4o-2024-08-06":
        model = "gpt-4o"
    
    if model not in TOKEN_PRICES:
        print(f"Warning: Model {model} not found in TOKEN_PRICES")
        return 0.0
    
    prices = TOKEN_PRICES[model]
    input_cost = (input_tokens / 1000000) * prices["input"]  # Convert to millions of tokens
    output_cost = (output_tokens / 1000000) * prices["output"]  # Convert to millions of tokens
    return input_cost + output_cost

def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare the dataset for analysis"""
    df = pd.read_csv("./LmSys_analysis_results/lmsys_adaptive_result.csv")
    
    # Recalculate costs for all rows
    df['cost'] = df.apply(lambda row: calculate_cost(row['model'], row['input_tokens'], row['output_tokens']), axis=1)
    
    return df

def create_cost_analysis_plots(df: pd.DataFrame) -> None:
    """Create various plots analyzing costs and token usage"""
    # Set the style
    plt.style.use('default')
    
    # Create a figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Cost per Model
    plt.subplot(2, 2, 1)
    cost_by_model = df.groupby('model')['cost'].mean().sort_values(ascending=False)
    sns.barplot(x=cost_by_model.index, y=cost_by_model.values)
    plt.xticks(rotation=45, ha='right')
    plt.title('Average Cost per Model')
    plt.xlabel('Model')
    plt.ylabel('Average Cost ($)')
    
    # 2. Token Usage Distribution
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df, x='input_tokens', y='output_tokens', hue='model', alpha=0.6)
    plt.title('Input vs Output Tokens by Model')
    plt.xlabel('Input Tokens')
    plt.ylabel('Output Tokens')
    
    # 3. Cost vs Token Usage
    plt.subplot(2, 2, 3)
    sns.scatterplot(data=df, x='input_tokens', y='cost', hue='model', alpha=0.6)
    plt.title('Cost vs Input Tokens')
    plt.xlabel('Input Tokens')
    plt.ylabel('Cost ($)')
    
    # 4. Token Usage Box Plot
    plt.subplot(2, 2, 4)
    df_melted = pd.melt(df, id_vars=['model'], value_vars=['input_tokens', 'output_tokens'])
    sns.boxplot(data=df_melted, x='model', y='value', hue='variable')
    plt.xticks(rotation=45, ha='right')
    plt.title('Token Usage Distribution by Model')
    plt.xlabel('Model')
    plt.ylabel('Number of Tokens')
    
    plt.tight_layout()
    plt.savefig('./LmSys_analysis_results/cost_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_statistics(df: pd.DataFrame) -> None:
    """Create summary statistics and save to a text file"""
    summary = []
    
    # Overall statistics
    summary.append("=== Overall Statistics ===")
    summary.append(f"Total number of entries: {len(df)}")
    summary.append(f"Total cost: ${df['cost'].sum():.4f}")
    summary.append(f"Average cost per entry: ${df['cost'].mean():.4f}")
    summary.append("\n")
    
    # Per model statistics
    summary.append("=== Per Model Statistics ===")
    model_stats = df.groupby('model').agg({
        'cost': ['count', 'sum', 'mean'],
        'input_tokens': ['mean', 'std'],
        'output_tokens': ['mean', 'std']
    }).round(4)
    
    summary.append(model_stats.to_string())
    
    # Save to file
    with open('./LmSys_analysis_results/summary_statistics.txt', 'w') as f:
        f.write('\n'.join(summary))

def main() -> None:
    # Load data
    df = load_and_prepare_data()
    
    # Create visualizations
    create_cost_analysis_plots(df)
    
    # Create summary statistics
    create_summary_statistics(df)
    
    print("Analysis complete! Check the following files:")
    print("1. ./LmSys_analysis_results/cost_analysis.png")
    print("2. ./LmSys_analysis_results/summary_statistics.txt")

if __name__ == "__main__":
    main() 