#!/usr/bin/env python3
"""
GenAI-Perf Results Analysis Script for Go LLM API
Analyzes benchmark results and generates performance reports and visualizations.
"""

import pandas as pd
import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime
import numpy as np

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class GenAIPerfAnalyzer:
    def __init__(self, results_dir='./results'):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
    def parse_genai_perf_csv(self, csv_path):
        """Parse GenAI-Perf CSV results file"""
        try:
            # Read the CSV file, skipping metadata rows
            df = pd.read_csv(csv_path, skiprows=8)
            
            # Extract key metrics
            metrics = {}
            for _, row in df.iterrows():
                metric_name = row['Metric']
                if metric_name == 'Throughput':
                    metrics['throughput_tps'] = row['Mean']
                elif metric_name == 'Time to First Token':
                    metrics['ttft_ms'] = row['Mean']
                elif metric_name == 'Inter Token Latency':
                    metrics['itl_ms'] = row['Mean']
                elif metric_name == 'Request Latency':
                    metrics['e2e_latency_ms'] = row['Mean']
                elif metric_name == 'Time Per Output Token':
                    metrics['time_per_output_token_ms'] = row['Mean']
                    
            return metrics
        except Exception as e:
            print(f"Error parsing {csv_path}: {e}")
            return None
    
    def parse_genai_perf_json(self, json_path):
        """Parse GenAI-Perf JSON results file"""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Extract metrics from JSON format
            metrics = {}
            if 'service_kind' in data:
                # New format
                for key, value in data.items():
                    if key == 'request_throughput':
                        metrics['throughput_tps'] = value
                    elif key == 'time_to_first_token':
                        metrics['ttft_ms'] = value.get('mean', 0)
                    elif key == 'inter_token_latency':
                        metrics['itl_ms'] = value.get('mean', 0)
                    elif key == 'request_latency':
                        metrics['e2e_latency_ms'] = value.get('mean', 0)
            else:
                # Legacy format
                metrics = data
                
            return metrics
        except Exception as e:
            print(f"Error parsing {json_path}: {e}")
            return None
    
    def extract_test_info(self, filepath):
        """Extract test information from file path"""
        path_parts = Path(filepath).parts
        filename = Path(filepath).stem
        
        # Try to extract concurrency from filename
        concurrency = 1
        if '_c' in filename:
            try:
                concurrency = int(filename.split('_c')[-1])
            except ValueError:
                pass
        
        # Try to extract test type from path or filename
        test_type = 'unknown'
        for part in path_parts:
            if any(keyword in part.lower() for keyword in ['quick', 'code', 'text', 'question', 'long']):
                test_type = part
                break
        
        if test_type == 'unknown' and '_' in filename:
            test_type = filename.split('_')[0]
            
        return {'concurrency': concurrency, 'test_type': test_type}
    
    def scan_results_directory(self):
        """Scan results directory for benchmark files"""
        results = []
        
        # Look for CSV files first
        csv_files = list(self.results_dir.rglob('*_genai_perf.csv'))
        for csv_file in csv_files:
            metrics = self.parse_genai_perf_csv(csv_file)
            if metrics:
                test_info = self.extract_test_info(csv_file)
                result = {**metrics, **test_info, 'source_file': str(csv_file)}
                results.append(result)
        
        # Look for JSON files
        json_files = list(self.results_dir.rglob('*.json'))
        for json_file in json_files:
            if 'genai_perf' not in json_file.stem:  # Skip if already processed as CSV
                metrics = self.parse_genai_perf_json(json_file)
                if metrics:
                    test_info = self.extract_test_info(json_file)
                    result = {**metrics, **test_info, 'source_file': str(json_file)}
                    results.append(result)
        
        return pd.DataFrame(results)
    
    def generate_performance_plots(self, df):
        """Generate comprehensive performance visualization plots"""
        if df.empty:
            print("No data to plot")
            return
        
        # 1. Throughput vs Concurrency
        plt.figure(figsize=(12, 8))
        if 'test_type' in df.columns and len(df['test_type'].unique()) > 1:
            for test_type in df['test_type'].unique():
                test_data = df[df['test_type'] == test_type]
                if not test_data.empty:
                    plt.plot(test_data['concurrency'], test_data['throughput_tps'], 
                            'o-', label=test_type, linewidth=2, markersize=8)
        else:
            plt.plot(df['concurrency'], df['throughput_tps'], 'o-', linewidth=2, markersize=8)
        
        plt.xlabel('Concurrency Level', fontsize=12)
        plt.ylabel('Throughput (tokens/second)', fontsize=12)
        plt.title('Go LLM API: Throughput vs Concurrency', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'throughput_vs_concurrency.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Latency Metrics
        plt.figure(figsize=(15, 5))
        
        # TTFT
        plt.subplot(1, 3, 1)
        plt.plot(df['concurrency'], df['ttft_ms'], 'o-', color='red', linewidth=2, markersize=8)
        plt.xlabel('Concurrency')
        plt.ylabel('Time to First Token (ms)')
        plt.title('TTFT vs Concurrency')
        plt.grid(True, alpha=0.3)
        
        # End-to-End Latency
        plt.subplot(1, 3, 2)
        plt.plot(df['concurrency'], df['e2e_latency_ms'], 'o-', color='blue', linewidth=2, markersize=8)
        plt.xlabel('Concurrency')
        plt.ylabel('End-to-End Latency (ms)')
        plt.title('E2E Latency vs Concurrency')
        plt.grid(True, alpha=0.3)
        
        # Inter Token Latency
        plt.subplot(1, 3, 3)
        if 'itl_ms' in df.columns:
            plt.plot(df['concurrency'], df['itl_ms'], 'o-', color='green', linewidth=2, markersize=8)
            plt.xlabel('Concurrency')
            plt.ylabel('Inter Token Latency (ms)')
            plt.title('ITL vs Concurrency')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'latency_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Latency vs Throughput Trade-off
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df['ttft_ms'], df['throughput_tps'], 
                             c=df['concurrency'], cmap='viridis', 
                             s=100, alpha=0.7, edgecolors='black')
        plt.colorbar(scatter, label='Concurrency Level')
        plt.xlabel('Time to First Token (ms)', fontsize=12)
        plt.ylabel('Throughput (tokens/second)', fontsize=12)
        plt.title('Go LLM API: Latency vs Throughput Trade-off', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Annotate points with concurrency levels
        for i, row in df.iterrows():
            plt.annotate(f'C={row["concurrency"]}', 
                        (row['ttft_ms'], row['throughput_tps']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'latency_vs_throughput.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Performance Summary Heatmap
        if len(df) > 3:
            plt.figure(figsize=(12, 8))
            
            # Create pivot table for heatmap
            pivot_data = df.pivot_table(
                values='throughput_tps', 
                index='test_type', 
                columns='concurrency', 
                aggfunc='mean'
            )
            
            sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Throughput (tokens/second)'})
            plt.title('Go LLM API: Performance Heatmap', fontsize=14, fontweight='bold')
            plt.xlabel('Concurrency Level', fontsize=12)
            plt.ylabel('Test Type', fontsize=12)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Performance plots saved to {self.plots_dir}/")
    
    def generate_summary_report(self, df):
        """Generate a comprehensive summary report"""
        if df.empty:
            print("No data available for summary report")
            return
        
        report_path = self.results_dir / 'benchmark_summary_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("Go LLM API Benchmark Summary Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Test Runs: {len(df)}\n\n")
            
            # Overall Statistics
            f.write("Overall Performance Statistics:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Max Throughput: {df['throughput_tps'].max():.2f} tokens/second\n")
            f.write(f"Min TTFT: {df['ttft_ms'].min():.2f} ms\n")
            f.write(f"Min E2E Latency: {df['e2e_latency_ms'].min():.2f} ms\n")
            f.write(f"Optimal Concurrency: {df.loc[df['throughput_tps'].idxmax(), 'concurrency']}\n\n")
            
            # Performance by Test Type
            if 'test_type' in df.columns:
                f.write("Performance by Test Type:\n")
                f.write("-" * 30 + "\n")
                for test_type in df['test_type'].unique():
                    test_data = df[df['test_type'] == test_type]
                    f.write(f"\n{test_type}:\n")
                    f.write(f"  Max Throughput: {test_data['throughput_tps'].max():.2f} tokens/second\n")
                    f.write(f"  Min TTFT: {test_data['ttft_ms'].min():.2f} ms\n")
                    f.write(f"  Avg E2E Latency: {test_data['e2e_latency_ms'].mean():.2f} ms\n")
            
            # Concurrency Analysis
            f.write("\nConcurrency Analysis:\n")
            f.write("-" * 20 + "\n")
            concurrency_stats = df.groupby('concurrency').agg({
                'throughput_tps': 'mean',
                'ttft_ms': 'mean',
                'e2e_latency_ms': 'mean'
            }).round(2)
            
            for concurrency, stats in concurrency_stats.iterrows():
                f.write(f"Concurrency {concurrency}:\n")
                f.write(f"  Throughput: {stats['throughput_tps']:.2f} tokens/second\n")
                f.write(f"  TTFT: {stats['ttft_ms']:.2f} ms\n")
                f.write(f"  E2E Latency: {stats['e2e_latency_ms']:.2f} ms\n\n")
            
            # Recommendations
            f.write("Recommendations:\n")
            f.write("-" * 15 + "\n")
            
            optimal_concurrency = df.loc[df['throughput_tps'].idxmax(), 'concurrency']
            f.write(f"• Optimal concurrency level: {optimal_concurrency}\n")
            
            if df['ttft_ms'].max() > 1000:
                f.write("• Consider optimizing Time to First Token (>1000ms detected)\n")
            
            if df['throughput_tps'].max() < 10:
                f.write("• Low throughput detected - consider scaling improvements\n")
            
            f.write("• Monitor performance degradation at high concurrency levels\n")
        
        print(f"✓ Summary report saved to {report_path}")
    
    def save_detailed_results(self, df):
        """Save detailed results to CSV"""
        if df.empty:
            print("No data to save")
            return
        
        csv_path = self.results_dir / 'go_api_benchmark_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Detailed results saved to {csv_path}")
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting GenAI-Perf Results Analysis...")
        print(f"Results directory: {self.results_dir}")
        
        # Scan for results
        df = self.scan_results_directory()
        
        if df.empty:
            print("No benchmark results found!")
            print("Make sure you have run benchmarks first using:")
            print("  docker-compose exec genai-perf bash scripts/benchmark_go_api.sh")
            return
        
        print(f"Found {len(df)} benchmark results")
        
        # Display basic stats
        print("\nBasic Statistics:")
        print(f"Throughput range: {df['throughput_tps'].min():.2f} - {df['throughput_tps'].max():.2f} tokens/second")
        print(f"TTFT range: {df['ttft_ms'].min():.2f} - {df['ttft_ms'].max():.2f} ms")
        print(f"Concurrency levels tested: {sorted(df['concurrency'].unique())}")
        
        # Generate outputs
        self.generate_performance_plots(df)
        self.generate_summary_report(df)
        self.save_detailed_results(df)
        
        print("\n" + "="*50)
        print("Analysis Complete!")
        print("="*50)
        print(f"📊 Plots: {self.plots_dir}/")
        print(f"📄 Summary: {self.results_dir}/benchmark_summary_report.txt")
        print(f"📊 CSV Data: {self.results_dir}/go_api_benchmark_results.csv")

def main():
    parser = argparse.ArgumentParser(description='Analyze GenAI-Perf benchmark results')
    parser.add_argument('--results-dir', default='./results', 
                       help='Directory containing benchmark results')
    parser.add_argument('--plots-only', action='store_true',
                       help='Generate plots only')
    parser.add_argument('--summary-only', action='store_true',
                       help='Generate summary report only')
    
    args = parser.parse_args()
    
    analyzer = GenAIPerfAnalyzer(args.results_dir)
    
    if args.plots_only:
        df = analyzer.scan_results_directory()
        analyzer.generate_performance_plots(df)
    elif args.summary_only:
        df = analyzer.scan_results_directory()
        analyzer.generate_summary_report(df)
    else:
        analyzer.run_analysis()

if __name__ == "__main__":
    main()