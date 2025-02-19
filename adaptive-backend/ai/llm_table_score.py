import pandas as pd

class LLMTable:
    def __init__(self):
        # Define the data as a dictionary
        self.data = {
            "LLM Model": [
                "OpenAI GPT‑4", 
                "OpenAI o1", 
                "Anthropic Claude 3", 
                "Google Gemini Ultra", 
                "Mistral 7B", 
                "Llama‑2 7B", 
                "GPT‑J", 
                "Reka Core"
            ],
            "Linguistic": [95, 93, 92, 91, 80, 78, 72, 83],
            "Reasoning": [90, 92, 88, 89, 75, 73, 70, 80],
            "Retrieval": [88, 87, 85, 86, 78, 76, 68, 82],
            "Pragmatic": [92, 90, 90, 88, 80, 78, 70, 84],
            "Medicine": [85, 86, 84, 83, 74, 72, 67, 80],
            "Coding": [90, 88, 86, 87, 76, 75, 69, 81],
            "Science": [87, 89, 85, 86, 75, 73, 68, 80]
        }
        # Create the DataFrame
        self.llm_df = pd.DataFrame(self.data)

    def get_table(self):
        """Return the DataFrame containing the LLM benchmark data."""
        return self.llm_df

# Usage example:
if __name__ == "__main__":
    table = LLMTable()
    df = table.get_table()
    print(df)
