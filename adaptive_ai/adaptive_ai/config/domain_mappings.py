"""
Domain-based model mappings for minion protocol.
Maps domain types to task-specific model choices for minion service.
Includes both core specialized domains and fallback mappings for additional domains.
"""

from adaptive_ai.models.llm_classification_models import DomainType

# Domain-based model mappings for minion protocol
minion_domains = {
    DomainType.BUSINESS_AND_INDUSTRIAL: "meta-llama/Llama-3.1-8B-Instruct",
    DomainType.HEALTH: "Qwen/Qwen3-8B-Base",
    DomainType.NEWS: "mistralai/Mistral-7B-Instruct-v0.2",
    DomainType.OTHERDOMAINS: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    DomainType.REAL_ESTATE: "meta-llama/Llama-3.1-8B-Instruct",
    DomainType.COMPUTERS_AND_ELECTRONICS: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    DomainType.INTERNET_AND_TELECOM: "Qwen/Qwen3-8B-Base",
    DomainType.FINANCE: "meta-llama/Llama-3.1-8B-Instruct",
    DomainType.SCIENCE: "Qwen/Qwen3-8B-Base",
    DomainType.JOBS_AND_EDUCATION: "mistralai/Mistral-7B-Instruct-v0.2",
    DomainType.LAW_AND_GOVERNMENT: "meta-llama/Llama-3.1-8B-Instruct",
    DomainType.SENSITIVE_SUBJECTS: "meta-llama/Llama-3.1-8B-Instruct",
}

# Add missing domains that should use OTHERDOMAINS models
for domain in [
    DomainType.AUTOS_AND_VEHICLES,
    DomainType.BOOKS_AND_LITERATURE,
    DomainType.ADULT,
    DomainType.ONLINE_COMMUNITIES,
    DomainType.TRAVEL_AND_TRANSPORTATION,
    DomainType.SHOPPING,
    DomainType.ARTS_AND_ENTERTAINMENT,
    DomainType.HOME_AND_GARDEN,
    DomainType.HOBBIES_AND_LEISURE,
    DomainType.PEOPLE_AND_SOCIETY,
    DomainType.FOOD_AND_DRINK,
    DomainType.SPORTS,
    DomainType.BEAUTY_AND_FITNESS,
    DomainType.PETS_AND_ANIMALS,
    DomainType.GAMES,
    DomainType.REFERENCE,
]:
    minion_domains[domain] = minion_domains[DomainType.OTHERDOMAINS]
