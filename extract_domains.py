#!/usr/bin/env python3
"""Extract and display the 2D domain-task mapping tables."""

import sys
import os

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import domain types and task types from the models
try:
    # Mock the missing dependencies
    class MockBaseModel:
        pass
    
    class MockField:
        def __init__(self, *args, **kwargs):
            pass
    
    # Inject mocks into sys.modules before importing
    import types
    pydantic_mock = types.ModuleType('pydantic')
    pydantic_mock.BaseModel = MockBaseModel
    pydantic_mock.Field = MockField
    sys.modules['pydantic'] = pydantic_mock
    
    # Now we can import our modules
    from adaptive_ai.models.llm_classification_models import DomainType
    from adaptive_ai.models.llm_core_models import TaskType
    
    print("# 2D DOMAIN-TASK MAPPING TABLES")
    print("=" * 80)
    
    # Define the domains and tasks
    domains = [
        "BUSINESS_AND_INDUSTRIAL",
        "COMPUTERS_AND_ELECTRONICS", 
        "FINANCE",
        "HEALTH",
        "INTERNET_AND_TELECOM",
        "JOBS_AND_EDUCATION",
        "LAW_AND_GOVERNMENT",
        "NEWS",
        "REAL_ESTATE",
        "SCIENCE",
        "SENSITIVE_SUBJECTS",
        "OTHERDOMAINS"
    ]
    
    tasks = [
        "CODE_GENERATION",
        "OPEN_QA",
        "SUMMARIZATION",
        "TEXT_GENERATION",
        "CHATBOT",
        "CLASSIFICATION",
        "CLOSED_QA",
        "REWRITE",
        "BRAINSTORMING",
        "EXTRACTION",
        "OTHER"
    ]
    
    # Print header
    print("\n## STANDARD PROTOCOL (Remote LLM Models)")
    print("-" * 50)
    
    # Print table header
    header = "Domain".ljust(25) + " | "
    for task in tasks:
        header += task[:12].ljust(14)
    print(header)
    print("-" * len(header))
    
    # Read the actual configuration from the file
    standard_mapping = {}
    minion_mapping = {}
    
    with open('/Users/attaimen/gitrepos/adaptive/adaptive_ai/adaptive_ai/config/model_catalog.py', 'r') as f:
        content = f.read()
    
    # Extract domain mappings from the file content
    lines = content.split('\n')
    current_domain = None
    current_dict = None
    in_domains = False
    in_minion_domains = False
    
    for line in lines:
        line = line.strip()
        
        if 'domains: dict[DomainType, dict[TaskType, list[TaskModelEntry]]] = {' in line:
            in_domains = True
            in_minion_domains = False
            continue
        elif 'minion_domains: dict[DomainType, dict[TaskType, str]] = {' in line:
            in_domains = False
            in_minion_domains = True
            continue
        elif line == '}' and (in_domains or in_minion_domains):
            in_domains = False
            in_minion_domains = False
            continue
            
        if in_domains and line.startswith('DomainType.'):
            domain_name = line.split('.')[1].split(':')[0]
            current_domain = domain_name
            if current_domain not in standard_mapping:
                standard_mapping[current_domain] = {}
        elif in_minion_domains and line.startswith('DomainType.'):
            domain_name = line.split('.')[1].split(':')[0]
            current_domain = domain_name
            if current_domain not in minion_mapping:
                minion_mapping[current_domain] = {}
        elif current_domain and line.startswith('TaskType.'):
            task_name = line.split('.')[1].split(':')[0]
            if in_domains:
                # For standard, get the first model from the list
                if 'TaskModelEntry(' in content:
                    # Find the first model in this task section
                    next_lines = []
                    found_start = False
                    for next_line in lines[lines.index(line):]:
                        if 'TaskModelEntry(' in next_line:
                            model_match = next_line.split('model_name="')[1].split('"')[0] if 'model_name="' in next_line else "unknown"
                            standard_mapping[current_domain][task_name] = model_match
                            break
            elif in_minion_domains:
                # For minion, get the model name directly
                if '"' in line:
                    model_name = line.split('"')[1]
                    minion_mapping[current_domain][task_name] = model_name
    
    # Print standard protocol table
    for domain in domains:
        if domain in standard_mapping:
            row = domain.ljust(25) + " | "
            for task in tasks:
                model = standard_mapping[domain].get(task, "N/A")[:12]
                row += model.ljust(14)
            print(row)
    
    print("\n## MINION PROTOCOL (HuggingFace Specialist Models)")
    print("-" * 50)
    
    # Print table header
    header = "Domain".ljust(25) + " | "
    for task in tasks:
        header += task[:12].ljust(16)
    print(header)
    print("-" * len(header))
    
    # Print minion protocol table
    for domain in domains:
        if domain in minion_mapping:
            row = domain.ljust(25) + " | "
            for task in tasks:
                model = minion_mapping[domain].get(task, "N/A")
                # Shorten long model names
                if "/" in model:
                    model = model.split("/")[-1][:14]
                row += model.ljust(16)
            print(row)
    
    print(f"\n## Summary")
    print(f"- **Total Domains**: {len(domains)}")
    print(f"- **Total Tasks**: {len(tasks)}")
    print(f"- **Standard Protocol Coverage**: {len(standard_mapping)}/{len(domains)} domains")
    print(f"- **Minion Protocol Coverage**: {len(minion_mapping)}/{len(domains)} domains")
    print(f"- **Total Combinations**: {len(domains)} × {len(tasks)} = {len(domains) * len(tasks)} mappings per protocol")
    
except ImportError as e:
    print(f"Import error: {e}")
    print("Falling back to manual extraction...")
    
    # Manual extraction as fallback
    domains = [
        "BUSINESS_AND_INDUSTRIAL",
        "COMPUTERS_AND_ELECTRONICS", 
        "FINANCE",
        "HEALTH",
        "INTERNET_AND_TELECOM",
        "JOBS_AND_EDUCATION",
        "LAW_AND_GOVERNMENT",
        "NEWS",
        "REAL_ESTATE",
        "SCIENCE",
        "SENSITIVE_SUBJECTS",
        "OTHERDOMAINS"
    ]
    
    tasks = [
        "CODE_GENERATION",
        "OPEN_QA", 
        "SUMMARIZATION",
        "TEXT_GENERATION",
        "CHATBOT",
        "CLASSIFICATION",
        "CLOSED_QA",
        "REWRITE",
        "BRAINSTORMING",
        "EXTRACTION",
        "OTHER"
    ]
    
    print("# 2D DOMAIN-TASK MAPPING TABLES")
    print("=" * 80)
    print(f"\n**Consolidated to {len(domains)} domains × {len(tasks)} tasks = {len(domains) * len(tasks)} total mappings per protocol**")
    print(f"\n**Domains**: {', '.join(domains)}")
    print(f"\n**Tasks**: {', '.join(tasks)}")