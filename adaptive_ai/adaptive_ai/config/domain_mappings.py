"""
Domain-based model mappings for minion protocol.
Maps domain types to task-specific model choices for minion service.
Includes both core specialized domains and fallback mappings for additional domains.
"""

from adaptive_ai.models.llm_classification_models import DomainType

# Domain-based model mappings for minion protocol
minion_domains = {
    DomainType.BUSINESS_AND_INDUSTRIAL: "Qwen/Qwen2.5-7B-Instruct",
    DomainType.HEALTH: "microsoft/Phi-4-mini-reasoning",
    DomainType.NEWS: "microsoft/Phi-4-mini-reasoning",
    DomainType.OTHERDOMAINS: "microsoft/Phi-4-mini-reasoning",
    DomainType.REAL_ESTATE: "microsoft/Phi-4-mini-reasoning",
    DomainType.COMPUTERS_AND_ELECTRONICS: "deepseek-ai/deepseek-coder-6.7b-base",
    DomainType.INTERNET_AND_TELECOM: "microsoft/Phi-4-mini-reasoning",
    DomainType.FINANCE: "instruction-pretrain/finance-Llama3-8B",
    DomainType.SCIENCE: "Qwen/Qwen2.5-Math-7B-Instruct",
    DomainType.JOBS_AND_EDUCATION: "microsoft/Phi-4-mini-reasoning",
    DomainType.LAW_AND_GOVERNMENT: "ricdomolm/lawma-8b",
    DomainType.SENSITIVE_SUBJECTS: "meta-llama/Meta-Llama-3-8B-Instruct",
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
