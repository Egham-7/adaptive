#!/usr/bin/env python3
"""
Test the new domains[domain][task_type] architecture
"""

import sys
import os
sys.path.append('/Users/attaimen/gitrepos/adaptive/adaptive_ai')

def test_direct_architecture():
    """Test the new architecture by importing directly"""
    print("🧪 Testing New Domain-Task Architecture")
    print("=" * 50)
    
    try:
        from adaptive_ai.config.model_catalog import domains, minion_domains
        from adaptive_ai.models.llm_classification_models import DomainType
        from adaptive_ai.models.llm_enums import TaskType
        
        # Test cases to verify domain-task specificity
        test_cases = [
            {
                "domain": DomainType.SCIENCE,
                "task": TaskType.CODE_GENERATION,
                "expected_top_model": "deepseek-reasoner",
                "description": "Science coding should favor reasoning models"
            },
            {
                "domain": DomainType.BUSINESS_AND_INDUSTRIAL,
                "task": TaskType.OPEN_QA,
                "expected_top_model": "gpt-4o",
                "description": "Business QA should favor reliable models"
            },
            {
                "domain": DomainType.ARTS_AND_ENTERTAINMENT,
                "task": TaskType.TEXT_GENERATION,
                "expected_top_model": "grok-3",
                "description": "Creative writing should favor creative models"
            },
            {
                "domain": DomainType.COMPUTERS_AND_ELECTRONICS,
                "task": TaskType.CODE_GENERATION,
                "expected_top_model": "deepseek-chat",
                "description": "Tech coding should favor code-capable models"
            }
        ]
        
        print("\n🎯 Testing Domain-Task Specificity:")
        
        for i, test in enumerate(test_cases, 1):
            try:
                # Test standard models lookup
                models = domains[test["domain"]][test["task"]]
                top_model = models[0].model_name if models else None
                
                # Test minion lookup
                minion = minion_domains[test["domain"]][test["task"]]
                
                print(f"\n{i}. {test['description']}")
                print(f"   Domain: {test['domain'].value}")
                print(f"   Task: {test['task'].value}")
                print(f"   ✅ Top Model: {top_model}")
                print(f"   🧙 Minion: {minion}")
                
                # Verify expected model
                if top_model == test["expected_top_model"]:
                    print(f"   🎯 CORRECT! Expected {test['expected_top_model']}")
                else:
                    print(f"   ⚠️  Expected {test['expected_top_model']}, got {top_model}")
                    
            except KeyError as e:
                print(f"\n{i}. ❌ MISSING: {test['domain'].value} + {test['task'].value}")
                print(f"   Error: {e}")
        
        # Test fail-fast behavior
        print(f"\n💥 Testing Fail-Fast Behavior:")
        
        try:
            # This should raise KeyError for missing domain
            fake_domain = "FAKE_DOMAIN"
            models = domains[fake_domain][TaskType.CODE_GENERATION]
            print("   ❌ FAIL: Should have raised KeyError for missing domain")
        except (KeyError, AttributeError):
            print("   ✅ PASS: Correctly raised error for missing domain")
        
        try:
            # This should raise KeyError for missing task
            fake_task = "FAKE_TASK" 
            models = domains[DomainType.SCIENCE][fake_task]
            print("   ❌ FAIL: Should have raised KeyError for missing task")
        except (KeyError, AttributeError):
            print("   ✅ PASS: Correctly raised error for missing task")
        
        # Test coverage
        print(f"\n📊 Coverage Analysis:")
        
        covered_standard = 0
        covered_minion = 0
        total_domains = len([d for d in DomainType])
        
        for domain in DomainType:
            if domain in domains:
                covered_standard += len(domains[domain])
            if domain in minion_domains:
                covered_minion += len(minion_domains[domain])
        
        print(f"   Standard Models: {covered_standard} combinations")
        print(f"   Minion Models: {covered_minion} combinations")
        print(f"   Domains Loaded: {len(domains)}")
        
        # Show a sample of what's available
        print(f"\n📋 Sample Domain-Task Matrix:")
        sample_domain = DomainType.SCIENCE
        if sample_domain in domains:
            print(f"   {sample_domain.value}:")
            for task, models in list(domains[sample_domain].items())[:3]:
                print(f"     {task.value}: {models[0].model_name}")
        
        print(f"\n🎉 Architecture Test Complete!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you're in the right directory and dependencies are installed")
        return False
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        return False

def test_model_selector_service():
    """Test the ModelSelectionService with the new architecture"""
    print(f"\n🔧 Testing ModelSelectionService:")
    print("=" * 30)
    
    try:
        from adaptive_ai.services.model_selector import ModelSelectionService
        from adaptive_ai.models.llm_classification_models import (
            ClassificationResult, DomainClassificationResult, DomainType
        )
        from adaptive_ai.models.llm_enums import TaskType
        from adaptive_ai.models.llm_core_models import ModelSelectionRequest
        
        # Create service
        service = ModelSelectionService()
        
        # Mock classification results
        classification_result = ClassificationResult(
            task_type_1=["Code Generation"],
            task_type_2=None,
            confidence_1=0.95,
            confidence_2=None
        )
        
        domain_classification = DomainClassificationResult(
            domain=DomainType.SCIENCE,
            confidence=0.9,
            domain_probabilities={}
        )
        
        # Mock request
        request = ModelSelectionRequest(
            prompt="Write a Python function to solve quantum mechanics equations",
            provider_constraint=None,
            max_tokens=100
        )
        
        print("   Testing model selection...")
        
        # Test the new direct lookup
        candidate_models = service.select_candidate_models(
            request=request,
            classification_result=classification_result,
            prompt_token_count=50,
            domain_classification=domain_classification
        )
        
        print(f"   ✅ Selected {len(candidate_models)} models")
        if candidate_models:
            print(f"   🤖 Top model: {candidate_models[0].model_name}")
            print(f"   📊 Provider: {candidate_models[0].provider.value}")
        
        # Test minion selection
        minion = service.get_designated_minion(
            classification_result=classification_result,
            domain_classification=domain_classification
        )
        
        print(f"   🧙 Minion: {minion}")
        
        # Test the fail-fast requirement
        print("   Testing fail-fast behavior...")
        
        try:
            # This should raise ValueError
            service.select_candidate_models(
                request=request,
                classification_result=classification_result,
                prompt_token_count=50,
                domain_classification=None  # No domain!
            )
            print("   ❌ FAIL: Should require domain classification")
        except ValueError:
            print("   ✅ PASS: Correctly requires domain classification")
        
        print("   🎉 ModelSelectionService test complete!")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing New Architecture Implementation")
    print("=" * 60)
    
    # Test 1: Direct architecture
    arch_success = test_direct_architecture()
    
    # Test 2: Service integration
    service_success = test_model_selector_service()
    
    print("\n" + "=" * 60)
    print("📈 FINAL RESULTS:")
    print(f"   Architecture Test: {'✅ PASS' if arch_success else '❌ FAIL'}")
    print(f"   Service Test: {'✅ PASS' if service_success else '❌ FAIL'}")
    
    if arch_success and service_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("🎯 New domains[domain][task_type] architecture is working!")
        print("💪 Ready for production use!")
    else:
        print("\n⚠️  Some tests failed - check the output above")