#!/usr/bin/env python3
import yaml
import sys

def check_yaml(file_path):
    try:
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        print(f"✅ {file_path} - Valid YAML syntax")
        return True
    except yaml.YAMLError as e:
        print(f"❌ {file_path} - YAML Error:")
        print(f"   {e}")
        return False
    except Exception as e:
        print(f"❌ {file_path} - Error:")
        print(f"   {e}")
        return False

if __name__ == "__main__":
    files = [
        ".github/workflows/merge-to-main.yml",
        ".github/workflows/frontend-ci.yml", 
        ".github/workflows/backend-ci.yml",
        ".github/workflows/ai-service-ci.yml"
    ]
    
    all_valid = True
    for file_path in files:
        if not check_yaml(file_path):
            all_valid = False
    
    if all_valid:
        print("\n🎉 All YAML files are valid!")
        sys.exit(0)
    else:
        print("\n💥 Some YAML files have errors!")
        sys.exit(1)