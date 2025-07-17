"""
Domain-Task Matrix Generator
Automatically generates comprehensive domain-task model combinations.
"""

from adaptive_ai.models.llm_classification_models import DomainType
from adaptive_ai.models.llm_enums import TaskType
from adaptive_ai.models.llm_core_models import TaskModelEntry
from adaptive_ai.models.llm_enums import ProviderType


def generate_comprehensive_domain_task_matrix() -> dict[tuple[DomainType, TaskType], list[TaskModelEntry]]:
    """
    Generate a comprehensive domain-task matrix with intelligent model selection.
    
    Returns:
        Complete domain-task matrix with all combinations covered
    """
    matrix = {}
    
    # Define model preference templates by domain characteristics
    domain_templates = {
        # Technical domains - favor code-capable models
        "technical": [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
        
        # Scientific domains - favor reasoning models
        "scientific": [
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o1"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
        ],
        
        # Creative domains - favor creative models
        "creative": [
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        
        # Business domains - favor reliable models
        "business": [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
        ],
        
        # Fast domains - favor efficient models
        "fast": [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
            TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
        ],
        
        # Sensitive domains - favor careful models
        "sensitive": [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4.1"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"),
        ],
        
        # Educational domains - favor instruction-following models
        "educational": [
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-reasoner"),
            TaskModelEntry(provider=ProviderType.OPENAI, model_name="o3"),
        ],
    }
    
    # Map domains to templates
    domain_mappings = {
        DomainType.COMPUTERS_AND_ELECTRONICS: "technical",
        DomainType.SCIENCE: "scientific",
        DomainType.ARTS_AND_ENTERTAINMENT: "creative",
        DomainType.GAMES: "creative",
        DomainType.BUSINESS_AND_INDUSTRIAL: "business",
        DomainType.FINANCE: "business",
        DomainType.LAW_AND_GOVERNMENT: "business",
        DomainType.NEWS: "fast",
        DomainType.ONLINE_COMMUNITIES: "fast",
        DomainType.SHOPPING: "fast",
        DomainType.ADULT: "sensitive",
        DomainType.HEALTH: "sensitive",
        DomainType.SENSITIVE_SUBJECTS: "sensitive",
        DomainType.JOBS_AND_EDUCATION: "educational",
        DomainType.REFERENCE: "educational",
        DomainType.BOOKS_AND_LITERATURE: "educational",
        # Default to business for remaining domains
        DomainType.AUTOS_AND_VEHICLES: "business",
        DomainType.BEAUTY_AND_FITNESS: "fast",
        DomainType.FOOD_AND_DRINK: "fast",
        DomainType.HOBBIES_AND_LEISURE: "creative",
        DomainType.HOME_AND_GARDEN: "fast",
        DomainType.INTERNET_AND_TELECOM: "technical",
        DomainType.PEOPLE_AND_SOCIETY: "business",
        DomainType.PETS_AND_ANIMALS: "fast",
        DomainType.REAL_ESTATE: "business",
        DomainType.SPORTS: "fast",
        DomainType.TRAVEL_AND_TRANSPORTATION: "fast",
    }
    
    # Task-specific model adjustments
    task_adjustments = {
        TaskType.CODE_GENERATION: {
            "prefer": [
                TaskModelEntry(provider=ProviderType.DEEPSEEK, model_name="deepseek-chat"),
                TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            ],
            "avoid": []
        },
        TaskType.BRAINSTORMING: {
            "prefer": [
                TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3"),
                TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o"),
            ],
            "avoid": [TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini")]
        },
        TaskType.CLASSIFICATION: {
            "prefer": [
                TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
                TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            ],
            "avoid": []
        },
        TaskType.EXTRACTION: {
            "prefer": [
                TaskModelEntry(provider=ProviderType.OPENAI, model_name="gpt-4o-mini"),
                TaskModelEntry(provider=ProviderType.GROK, model_name="grok-3-mini"),
            ],
            "avoid": []
        },
    }
    
    # Generate all combinations
    for domain in DomainType:
        for task in TaskType:
            # Get base template
            template_name = domain_mappings.get(domain, "business")
            base_models = domain_templates[template_name].copy()
            
            # Apply task-specific adjustments
            if task in task_adjustments:
                adjustment = task_adjustments[task]
                # Add preferred models to front
                final_models = adjustment["prefer"] + base_models
                # Remove avoided models
                final_models = [m for m in final_models if m not in adjustment["avoid"]]
                # Remove duplicates while preserving order
                seen = set()
                unique_models = []
                for model in final_models:
                    key = (model.provider, model.model_name)
                    if key not in seen:
                        seen.add(key)
                        unique_models.append(model)
                final_models = unique_models
            else:
                final_models = base_models
            
            # Limit to top 3 models per combination
            matrix[(domain, task)] = final_models[:3]
    
    return matrix


def validate_matrix_coverage(matrix: dict[tuple[DomainType, TaskType], list[TaskModelEntry]]) -> dict:
    """
    Validate that the matrix covers all domain-task combinations.
    
    Args:
        matrix: The domain-task matrix to validate
        
    Returns:
        Validation report
    """
    total_combinations = len(DomainType) * len(TaskType)
    covered_combinations = len(matrix)
    
    missing_combinations = []
    for domain in DomainType:
        for task in TaskType:
            if (domain, task) not in matrix:
                missing_combinations.append((domain, task))
    
    # Check for empty model lists
    empty_combinations = []
    for key, models in matrix.items():
        if not models:
            empty_combinations.append(key)
    
    return {
        "total_combinations": total_combinations,
        "covered_combinations": covered_combinations,
        "coverage_percentage": (covered_combinations / total_combinations) * 100,
        "missing_combinations": missing_combinations,
        "empty_combinations": empty_combinations,
        "is_complete": len(missing_combinations) == 0 and len(empty_combinations) == 0
    }