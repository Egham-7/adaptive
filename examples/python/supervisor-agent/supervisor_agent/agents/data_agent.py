"""Data Agent - Handles data processing, analysis, and visualization tasks."""

import json
import tempfile
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from supervisor_agent.utils.config import get_config


@tool
def analyze_data(data: Union[str, List, Dict]) -> Dict[str, Any]:
    """Analyze data and provide statistical insights.
    
    Args:
        data: Data to analyze (JSON string, list, or dict)
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Convert data to pandas DataFrame if possible
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                # Try to parse as CSV-like string
                lines = data.strip().split('\n')
                if len(lines) > 1:
                    headers = lines[0].split(',')
                    rows = [line.split(',') for line in lines[1:]]
                    data = pd.DataFrame(rows, columns=headers)
                else:
                    return {"error": "Unable to parse data format"}
        
        if isinstance(data, (list, dict)):
            df = pd.DataFrame(data)
        else:
            df = data
        
        # Basic statistics
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        analysis = {
            "shape": df.shape,
            "columns": list(df.columns),
            "data_types": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Numeric analysis
        if len(numeric_columns) > 0:
            analysis["numeric_summary"] = df[numeric_columns].describe().to_dict()
        
        # Categorical analysis
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            analysis["categorical_summary"][col] = {
                "unique_values": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict()
            }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


@tool
def calculate_statistics(data: List[float], stat_type: str = "all") -> Dict[str, float]:
    """Calculate statistical measures for numerical data.
    
    Args:
        data: List of numerical values
        stat_type: Type of statistics ("all", "basic", "advanced")
        
    Returns:
        Dictionary with calculated statistics
    """
    try:
        arr = np.array(data)
        
        basic_stats = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(arr)
        }
        
        if stat_type == "basic":
            return basic_stats
        
        advanced_stats = {
            "variance": float(np.var(arr)),
            "q1": float(np.percentile(arr, 25)),
            "q3": float(np.percentile(arr, 75)),
            "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
            "skewness": float(pd.Series(arr).skew()),
            "kurtosis": float(pd.Series(arr).kurtosis())
        }
        
        if stat_type == "all":
            return {**basic_stats, **advanced_stats}
        else:
            return advanced_stats
            
    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}


@tool
def create_visualization(data: Union[str, List, Dict], chart_type: str, title: str = "Data Visualization") -> str:
    """Create a data visualization chart.
    
    Args:
        data: Data to visualize
        chart_type: Type of chart ("bar", "line", "scatter", "histogram", "pie")
        title: Chart title
        
    Returns:
        Path to the saved chart image
    """
    try:
        # Parse data
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return "Error: Unable to parse data"
        
        plt.figure(figsize=(10, 6))
        
        if chart_type.lower() == "bar":
            if isinstance(data, dict):
                plt.bar(data.keys(), data.values())
            elif isinstance(data, list) and all(isinstance(x, dict) for x in data):
                # Assume list of records
                df = pd.DataFrame(data)
                if len(df.columns) >= 2:
                    plt.bar(df.iloc[:, 0], df.iloc[:, 1])
            else:
                plt.bar(range(len(data)), data)
                
        elif chart_type.lower() == "line":
            if isinstance(data, dict):
                plt.plot(list(data.keys()), list(data.values()))
            else:
                plt.plot(data)
                
        elif chart_type.lower() == "scatter":
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (list, tuple)):
                x_vals = [point[0] for point in data]
                y_vals = [point[1] for point in data]
                plt.scatter(x_vals, y_vals)
            else:
                plt.scatter(range(len(data)), data)
                
        elif chart_type.lower() == "histogram":
            if isinstance(data, dict):
                plt.hist(list(data.values()))
            else:
                plt.hist(data)
                
        elif chart_type.lower() == "pie":
            if isinstance(data, dict):
                plt.pie(data.values(), labels=data.keys(), autopct='%1.1f%%')
            else:
                plt.pie(data, autopct='%1.1f%%')
        
        plt.title(title)
        plt.tight_layout()
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        plt.savefig(temp_file.name, dpi=150, bbox_inches='tight')
        plt.close()
        
        return temp_file.name
        
    except Exception as e:
        return f"Error creating visualization: {str(e)}"


@tool
def process_csv_data(csv_content: str, operation: str) -> Dict[str, Any]:
    """Process CSV data with various operations.
    
    Args:
        csv_content: CSV data as string
        operation: Operation to perform ("summary", "clean", "transform")
        
    Returns:
        Dictionary with processed results
    """
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(csv_content))
        
        if operation == "summary":
            return {
                "shape": df.shape,
                "columns": list(df.columns),
                "head": df.head().to_dict(),
                "info": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
            }
            
        elif operation == "clean":
            # Basic cleaning operations
            original_shape = df.shape
            df_cleaned = df.dropna()  # Remove null values
            df_cleaned = df_cleaned.drop_duplicates()  # Remove duplicates
            
            return {
                "original_shape": original_shape,
                "cleaned_shape": df_cleaned.shape,
                "removed_rows": original_shape[0] - df_cleaned.shape[0],
                "cleaned_data": df_cleaned.to_dict()
            }
            
        elif operation == "transform":
            # Basic transformations
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                
            return {
                "transformed_data": df.to_dict(),
                "transformations_applied": [
                    "Filled numeric nulls with mean values"
                ]
            }
            
    except Exception as e:
        return {"error": f"CSV processing failed: {str(e)}"}


@tool 
def generate_data_insights(data: Union[str, List, Dict]) -> str:
    """Generate natural language insights about the data using LLM.
    
    Args:
        data: Data to analyze
        
    Returns:
        Natural language insights about the data
    """
    config = get_config()
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        api_key=config.openai_api_key
    )
    
    # First analyze the data
    analysis = analyze_data(data)
    
    prompt = f"""Analyze this data analysis report and provide key insights:

{json.dumps(analysis, indent=2)}

Please provide:
1. Key findings and patterns
2. Data quality assessment
3. Interesting observations
4. Recommendations for further analysis
5. Potential data issues or anomalies

Write in clear, business-friendly language."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def perform_calculations(expression: str, variables: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Perform mathematical calculations safely.
    
    Args:
        expression: Mathematical expression to evaluate
        variables: Optional dictionary of variable values
        
    Returns:
        Dictionary with calculation results
    """
    try:
        # Safe evaluation environment
        safe_dict = {
            "__builtins__": {},
            "abs": abs,
            "max": max,
            "min": min,
            "sum": sum,
            "round": round,
            "pow": pow,
            "sqrt": np.sqrt,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "log": np.log,
            "exp": np.exp,
            "pi": np.pi,
            "e": np.e
        }
        
        if variables:
            safe_dict.update(variables)
        
        result = eval(expression, safe_dict)
        
        return {
            "expression": expression,
            "result": float(result) if isinstance(result, (int, float, np.number)) else result,
            "variables_used": variables or {}
        }
        
    except Exception as e:
        return {
            "error": f"Calculation failed: {str(e)}",
            "expression": expression
        }


class DataAgent:
    """Agent specialized in data processing and analysis tasks."""
    
    def __init__(self):
        """Initialize the Data Agent."""
        self.name = "Data Agent"
        self.description = "I handle data analysis, statistics, visualizations, and mathematical calculations."
        self.tools = [
            analyze_data,
            calculate_statistics,
            create_visualization,
            process_csv_data,
            generate_data_insights,
            perform_calculations,
        ]
        
        config = get_config()
        self.llm = ChatOpenAI(
            model=config.openai_model,
            temperature=config.temperature,
            api_key=config.openai_api_key
        )
    
    def get_system_message(self) -> str:
        """Get the system message for this agent."""
        return f"""You are the {self.name}. {self.description}

You have access to the following tools:
- analyze_data: Analyze datasets and provide statistical insights
- calculate_statistics: Calculate various statistical measures
- create_visualization: Generate charts and graphs
- process_csv_data: Handle CSV data processing operations
- generate_data_insights: Provide natural language insights about data
- perform_calculations: Execute mathematical calculations safely

When handling requests:
1. Identify the type of data analysis needed
2. Use appropriate tools to process and analyze the data
3. Provide clear, actionable insights
4. Create visualizations when helpful
5. Always validate data before processing

Only handle tasks related to data analysis, statistics, math, and visualizations. If asked about non-data topics, politely redirect to the supervisor."""
    
    def can_handle(self, message: str) -> bool:
        """Determine if this agent can handle the given message."""
        data_keywords = [
            "data", "analyze", "statistics", "chart", "graph", "plot", "csv",
            "calculate", "math", "number", "count", "average", "mean", "median",
            "sum", "total", "percentage", "ratio", "trend", "pattern", "insight",
            "visualization", "dashboard", "report", "metric", "measure", "formula",
            "equation", "probability", "correlation", "regression"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in data_keywords)