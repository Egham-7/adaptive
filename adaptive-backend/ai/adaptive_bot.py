from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ai.llms import domain_model_mapping, model_capabilities

def select_model(domain: str, prompt_complexity_score: float) -> str:
    if domain not in domain_model_mapping:
        raise ValueError(f"Domain '{domain}' is not recognized.")
    
    # Filter models suitable for the given domain
    suitable_models = domain_model_mapping[domain]
    
    # Find a model within the suitable models that matches the complexity score
    for model_name in suitable_models:
        complexity_range = model_capabilities[model_name]["complexity_range"]
        if complexity_range[0] <= prompt_complexity_score <= complexity_range[1]:
            return model_name
    
    # If no model matches the complexity score, return a default model
    return suitable_models[0]

# Step 4: ChatBot class that utilizes the selected model
class ChatBot:
    def __init__(self, domain: str, prompt_complexity_score: float, temperature: float = 0.2):
        # Select the appropriate model based on domain and complexity
        self.model_name = select_model(domain, prompt_complexity_score)
        # Initialize the ChatGroq model
        self.chat = ChatGroq(model_name=self.model_name, temperature=temperature)

    def send(self, text: str) -> str:
        system_message = "You are an expert"
        user_message = text
        prompt = ChatPromptTemplate.from_messages([("system", system_message), ("human", user_message)])
        response = (prompt | self.chat).invoke({"text": text})
        return {"ans":response.content,"model":self.model_name}

"""# Example usage
def main():
    domain = "Finance"
    prompt_complexity_score = 0.3  # Example complexity score
    bot = ChatBot(domain=domain, prompt_complexity_score=prompt_complexity_score)
    result = bot.send("Can you provide an analysis of the current stock market trends?")
    print(result)

if __name__ == "__main__":
    main()"""