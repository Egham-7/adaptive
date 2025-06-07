import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
from typing import Dict, List, Tuple
from tqdm import tqdm

# Configuration
TOKEN_PRICES = {
    'o3': {'input': 10.00, 'output': 40.00},
    'o4-mini': {'input': 1.1, 'output': 4.40},
    'gpt-4o': {'input': 5.00, 'output': 20.00},
    'gpt-4.1-mini': {'input': 0.40, 'output': 1.60},
    'gpt-4.1-nano': {'input': 0.10, 'output': 0.40},
    'gpt-4.1': {'input': 2.00, 'output': 2.00},
    'gemini-2.0-flash': {'input': 0.10, 'output': 0.40},
    'gemini-2.0-flash-lite': {'input': 0.075, 'output': 0.30},
    'deepseek-chat': {'input': 0.07, 'output': 1.10},
    'deepseek-reasoner': {'input': 0.14, 'output': 2.19},
    'claude-sonnet-4-0': {'input': 3.00, 'output': 15.00},
    'claude-3-5-haiku-latest': {'input': 0.80, 'output': 4.00},
    'claude-opus-4-0': {'input': 15.00, 'output': 75.00}
}

class CostAnalysisPipeline:
    def __init__(self):
        self.prediction_df = None
        self.token_counts_df = None
        self.combined_df = None
        self.costs_df = None

    def get_prediction(self, prompt: str) -> Dict:
        """Get model prediction for a given prompt."""
        if pd.isna(prompt):
            return {'model': None, 'provider': None, 'task_type': None}
        
        url = "http://localhost:8000/predict"
        payload = {"prompt": str(prompt)}
        response = requests.post(url, json=payload)
        return response.json()

    def process_predictions(self, input_file: str = 'token_counts.csv', limit: int = 10000) -> None:
        """Process predictions for the input data."""
        # Read and limit the input data
        self.token_counts_df = pd.read_csv(input_file).head(limit)
        
        # Get predictions for each prompt
        results = []
        print("Getting predictions for prompts...")
        for _, row in tqdm(self.token_counts_df.iterrows(), total=len(self.token_counts_df)):
            prediction = self.get_prediction(row['input_prompt'])
            results.append({
                'model': prediction.get('selected_model'),
                'provider': prediction.get('provider'),
                'task_type': prediction.get('task_type'),
                'original_prompt': row['input_prompt']
            })
        
        self.prediction_df = pd.DataFrame(results)
        self.prediction_df.to_csv('./prediction_results.csv', index=False)
        
        # Combine with token counts
        self.combined_df = self.prediction_df.copy()
        self.combined_df['input_tokens'] = self.token_counts_df['input_tokens']
        self.combined_df['output_tokens'] = self.token_counts_df['output_tokens']
        self.combined_df['total_tokens'] = self.combined_df['input_tokens'] + self.combined_df['output_tokens']
        self.combined_df.to_csv('./prediction_results_with_tokens.csv', index=False)

    def calculate_costs(self) -> None:
        """Calculate costs for each prediction."""
        def calculate_cost(row):
            model = row['model']
            if model not in TOKEN_PRICES:
                return 0.0
            input_cost = (row['input_tokens'] / 1_000_000) * TOKEN_PRICES[model]['input']
            output_cost = (row['output_tokens'] / 1_000_000) * TOKEN_PRICES[model]['output']
            return input_cost + output_cost

        self.costs_df = self.combined_df.copy()
        self.costs_df['cost'] = self.costs_df.apply(calculate_cost, axis=1)
        self.costs_df.to_csv('./prediction_results_with_costs.csv', index=False)

    def calculate_single_model_cost(self, model_name: str) -> float:
        """Calculate cost if using only one model for all requests."""
        if model_name not in TOKEN_PRICES:
            return 0.0
        
        total_input_tokens = self.costs_df['input_tokens'].sum()
        total_output_tokens = self.costs_df['output_tokens'].sum()
        
        input_cost = (total_input_tokens / 1_000_000) * TOKEN_PRICES[model_name]['input']
        output_cost = (total_output_tokens / 1_000_000) * TOKEN_PRICES[model_name]['output']
        return input_cost + output_cost

    def generate_visualizations(self) -> None:
        """Generate comprehensive visualizations of the cost analysis."""
        plt.style.use('ggplot')
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)

        # 1. Cost Comparison: Adaptive vs Single Models
        ax1 = fig.add_subplot(gs[0, :])
        single_model_costs = {model: self.calculate_single_model_cost(model) 
                            for model in TOKEN_PRICES.keys()}
        adaptive_cost = self.costs_df['cost'].sum()

        sorted_models = sorted(single_model_costs.items(), key=lambda x: x[1])
        model_names = [model for model, _ in sorted_models]
        single_costs = [cost for _, cost in sorted_models]

        bars = ax1.bar(model_names, single_costs, color='skyblue', alpha=0.7)
        ax1.axhline(y=adaptive_cost, color='red', linestyle='--', label='Adaptive Model Cost')

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}',
                    ha='center', va='bottom')

        ax1.set_title('Cost Comparison: Adaptive vs Single Model Scenarios', pad=20)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('Cost ($)')
        ax1.legend()
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. Cost Breakdown by Task Type
        ax2 = fig.add_subplot(gs[1, 0])
        task_costs = self.costs_df.groupby('task_type')['cost'].sum().sort_values(ascending=False)
        bars2 = ax2.bar(task_costs.index, task_costs.values, color='lightgreen', alpha=0.7)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}',
                    ha='center', va='bottom')

        ax2.set_title('Cost Breakdown by Task Type', pad=20)
        ax2.set_xlabel('Task Type')
        ax2.set_ylabel('Cost ($)')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. Cost per Token by Model
        ax3 = fig.add_subplot(gs[1, 1])
        model_token_costs = self.costs_df.groupby('model').agg({
            'cost': 'sum',
            'total_tokens': 'sum'
        }).reset_index()
        model_token_costs['cost_per_token'] = model_token_costs['cost'] / model_token_costs['total_tokens'] * 1000
        model_token_costs = model_token_costs.sort_values('cost_per_token', ascending=False)

        bars3 = ax3.bar(model_token_costs['model'], model_token_costs['cost_per_token'], 
                       color='salmon', alpha=0.7)
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.3f}',
                    ha='center', va='bottom')

        ax3.set_title('Cost per 1000 Tokens by Model', pad=20)
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Cost per 1000 Tokens ($)')
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 4. Cost Efficiency Comparison
        ax4 = fig.add_subplot(gs[2, :])
        efficiency_data = []
        for model in TOKEN_PRICES.keys():
            single_cost = self.calculate_single_model_cost(model)
            efficiency = (single_cost - adaptive_cost) / single_cost * 100
            efficiency_data.append((model, efficiency))

        efficiency_data.sort(key=lambda x: x[1], reverse=True)
        models = [x[0] for x in efficiency_data]
        efficiencies = [x[1] for x in efficiency_data]

        bars4 = ax4.bar(models, efficiencies, color='lightblue', alpha=0.7)
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')

        ax4.set_title('Cost Savings Using Adaptive Model Selection', pad=20)
        ax4.set_xlabel('Model')
        ax4.set_ylabel('Cost Savings (%)')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig('./cost_analysis.png', dpi=300, bbox_inches='tight')

    def print_analysis(self) -> None:
        """Print comprehensive analysis of the results."""
        print("\n=== Cost Analysis ===")

        # Cost by model
        print("\nCost Statistics by Model:")
        model_costs = self.costs_df.groupby('model').agg({
            'cost': ['sum', 'mean', 'count'],
            'input_tokens': 'sum',
            'output_tokens': 'sum'
        }).round(4)
        print(model_costs)

        # Cost by provider
        print("\nCost Statistics by Provider:")
        provider_costs = self.costs_df.groupby('provider').agg({
            'cost': ['sum', 'mean', 'count'],
            'input_tokens': 'sum',
            'output_tokens': 'sum'
        }).round(4)
        print(provider_costs)

        # Cost by task type
        print("\nCost Statistics by Task Type:")
        task_costs = self.costs_df.groupby('task_type').agg({
            'cost': ['sum', 'mean', 'count'],
            'input_tokens': 'sum',
            'output_tokens': 'sum'
        }).round(4)
        print(task_costs)

        # Overall statistics
        print("\n=== Overall Cost Statistics ===")
        print(f"Total cost: ${self.costs_df['cost'].sum():.4f}")
        print(f"Average cost per request: ${self.costs_df['cost'].mean():.4f}")
        print(f"Total input tokens: {self.costs_df['input_tokens'].sum():,}")
        print(f"Total output tokens: {self.costs_df['output_tokens'].sum():,}")
        print(f"Total tokens: {self.costs_df['total_tokens'].sum():,}")

        # Single model scenarios
        print("\n=== Cost Comparison: Single Model Scenarios ===")
        print("What if we used only one model for all requests:")
        for model in TOKEN_PRICES.keys():
            single_model_cost = self.calculate_single_model_cost(model)
            print(f"{model}: ${single_model_cost:.4f}")

def main():
    # Initialize and run the pipeline
    pipeline = CostAnalysisPipeline()
    
    # Process predictions
    print("Processing predictions...")
    pipeline.process_predictions()
    
    # Calculate costs
    print("Calculating costs...")
    pipeline.calculate_costs()
    
    # Generate visualizations
    print("Generating visualizations...")
    pipeline.generate_visualizations()
    
    # Print analysis
    print("Printing analysis...")
    pipeline.print_analysis()

if __name__ == "__main__":
    main() 