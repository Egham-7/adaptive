from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configuration
TOKEN_PRICES: Dict[str, Dict[str, float]] = {
    "o3": {"input": 10.00, "output": 40.00},
    "o4-mini": {"input": 1.1, "output": 4.40},
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1": {"input": 2.00, "output": 2.00},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "deepseek-chat": {"input": 0.07, "output": 1.10},
    "deepseek-reasoner": {"input": 0.14, "output": 2.19},
    "claude-sonnet-4-0": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
    "claude-opus-4-0": {"input": 15.00, "output": 75.00},
}


class CostAnalysisPipeline:
    def __init__(self) -> None:
        self.costs_df: pd.DataFrame | None = None

    def load_data(self, input_file: str = "/content/enhanced_dataset_10000_individual_requests.csv") -> None:
        """Load data with predictions already included"""
        print("Loading pre-predicted data...")
        self.costs_df = pd.read_csv(input_file)
        
        # Validate required columns
        required_cols = ['selected_model', 'provider', 'input_tokens', 'output_tokens']
        missing_cols = [col for col in required_cols if col not in self.costs_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        print(f"Successfully loaded {len(self.costs_df)} rows")
        print("Columns available:", list(self.costs_df.columns))

    def calculate_costs(self) -> None:
        """Calculate costs for each prediction."""
        if self.costs_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        def calculate_cost(row: pd.Series) -> float:
            model = str(row["selected_model"])
            if model not in TOKEN_PRICES:
                return 0.0
            input_cost = (float(row["input_tokens"]) / 1_000_000) * TOKEN_PRICES[model]["input"]
            output_cost = (float(row["output_tokens"]) / 1_000_000) * TOKEN_PRICES[model]["output"]
            return input_cost + output_cost

        print("Calculating costs...")
        self.costs_df["cost"] = self.costs_df.apply(calculate_cost, axis=1)
        self.costs_df.to_csv("./prediction_results_with_costs.csv", index=False)
        print("Cost calculations completed")

    def calculate_single_model_cost(self, model_name: str) -> float:
        """Calculate cost if using only one model for all requests."""
        if self.costs_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
            
        if model_name not in TOKEN_PRICES:
            return 0.0

        total_input_tokens = float(self.costs_df["input_tokens"].sum())
        total_output_tokens = float(self.costs_df["output_tokens"].sum())

        input_cost = (total_input_tokens / 1_000_000) * TOKEN_PRICES[model_name]["input"]
        output_cost = (total_output_tokens / 1_000_000) * TOKEN_PRICES[model_name]["output"]
        return input_cost + output_cost

    def generate_visualizations(self) -> None:
        """Generate comprehensive visualizations of the cost analysis."""
        if self.costs_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        plt.style.use("ggplot")
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)

        # 1. Cost Comparison: Adaptive vs Single Models
        ax1 = fig.add_subplot(gs[0, :])
        adaptive_cost = self.costs_df["cost"].sum()
        single_model_costs = {
            model: self.calculate_single_model_cost(model)
            for model in TOKEN_PRICES.keys()
        }

        sorted_models = sorted(single_model_costs.items(), key=lambda x: x[1])
        model_names = [model for model, _ in sorted_models]
        single_costs = np.array([cost for _, cost in sorted_models])

        bars = ax1.bar(model_names, single_costs, color="skyblue", alpha=0.7)
        ax1.axhline(y=adaptive_cost, color="red", linestyle="--", label="Adaptive Model Cost")

        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f"${height:.2f}", ha="center", va="bottom")

        ax1.set_title("Cost Comparison: Adaptive vs Single Model Scenarios", pad=20)
        ax1.set_xlabel("Model")
        ax1.set_ylabel("Cost ($)")
        ax1.legend()
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 2. Cost Breakdown by Task Type
        ax2 = fig.add_subplot(gs[1, 0])
        task_costs = self.costs_df.groupby("task_type")["cost"].sum().sort_values(ascending=False)
        bars2 = ax2.bar(task_costs.index, task_costs.values, color="lightgreen", alpha=0.7)

        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f"${height:.2f}", ha="center", va="bottom")

        ax2.set_title("Cost Breakdown by Task Type", pad=20)
        ax2.set_xlabel("Task Type")
        ax2.set_ylabel("Cost ($)")
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 3. Cost per Token by Model
        ax3 = fig.add_subplot(gs[1, 1])
        model_token_costs = (
            self.costs_df.groupby("selected_model")
            .agg({"cost": "sum", "input_tokens": "sum", "output_tokens": "sum"})
            .reset_index()
        )
        model_token_costs["total_tokens"] = model_token_costs["input_tokens"] + model_token_costs["output_tokens"]
        model_token_costs["cost_per_token"] = (
            model_token_costs["cost"] / model_token_costs["total_tokens"] * 1000
        )
        model_token_costs = model_token_costs.sort_values("cost_per_token", ascending=False)

        bars3 = ax3.bar(
            model_token_costs["selected_model"],
            model_token_costs["cost_per_token"],
            color="salmon",
            alpha=0.7,
        )
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f"${height:.3f}", ha="center", va="bottom")

        ax3.set_title("Cost per 1000 Tokens by Model", pad=20)
        ax3.set_xlabel("Model")
        ax3.set_ylabel("Cost per 1000 Tokens ($)")
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

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

        bars4 = ax4.bar(models, efficiencies, color="lightblue", alpha=0.7)
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2.0, height,
                    f"{height:.1f}%", ha="center", va="bottom")

        ax4.set_title("Cost Savings Using Adaptive Model Selection", pad=20)
        ax4.set_xlabel("Model")
        ax4.set_ylabel("Cost Savings (%)")
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()
        plt.savefig("./cost_analysis.png", dpi=300, bbox_inches="tight")
        print("Visualizations saved to cost_analysis.png")

    def print_analysis(self) -> None:
        """Print comprehensive analysis of the results."""
        if self.costs_df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n=== Cost Analysis ===")
        print(f"Total requests analyzed: {len(self.costs_df):,}")

        # Cost by model
        print("\nCost Statistics by Model:")
        model_stats = self.costs_df.groupby("selected_model").agg({
            "cost": ["sum", "mean", "count"],
            "input_tokens": "sum",
            "output_tokens": "sum"
        }).round(4)
        print(model_stats)

        # Cost by provider
        print("\nCost Statistics by Provider:")
        provider_stats = self.costs_df.groupby("provider").agg({
            "cost": ["sum", "mean", "count"],
            "input_tokens": "sum",
            "output_tokens": "sum"
        }).round(4)
        print(provider_stats)

        # Overall statistics
        print("\n=== Overall Cost Statistics ===")
        print(f"Total cost: ${self.costs_df['cost'].sum():.2f}")
        print(f"Average cost per request: ${self.costs_df['cost'].mean():.4f}")
        print(f"Total input tokens: {self.costs_df['input_tokens'].sum():,}")
        print(f"Total output tokens: {self.costs_df['output_tokens'].sum():,}")
        print(f"Total tokens: {self.costs_df['input_tokens'].sum() + self.costs_df['output_tokens'].sum():,}")

        # Single model scenarios
        print("\n=== Cost Comparison: Single Model Scenarios ===")
        for model in sorted(TOKEN_PRICES.keys()):
            cost = self.calculate_single_model_cost(model)
            print(f"{model:<25}: ${cost:>10.2f}")
            
        adaptive_cost = self.costs_df['cost'].sum()
        print(f"\n{'Adaptive selection':<25}: ${adaptive_cost:>10.2f}")
        print(f"Savings vs most expensive: {(max(self.calculate_single_model_cost(m) for m in TOKEN_PRICES) - adaptive_cost):.2f}")
        print(f"Savings vs cheapest: {(min(self.calculate_single_model_cost(m) for m in TOKEN_PRICES) - adaptive_cost):.2f}")


def main() -> None:
    pipeline = CostAnalysisPipeline()
    
    try:
        # Load pre-predicted data
        pipeline.load_data()
        
        # Calculate costs
        pipeline.calculate_costs()
        
        # Generate visualizations
        pipeline.generate_visualizations()
        
        # Print analysis
        pipeline.print_analysis()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()