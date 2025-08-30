"""Code Agent - Handles code generation, debugging, and testing tasks."""

import ast
import subprocess
import sys
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from supervisor_agent.utils.config import get_config


@tool
def generate_code(language: str, description: str) -> str:
    """Generate code in the specified language based on description.
    
    Args:
        language: Programming language (python, javascript, sql, etc.)
        description: Description of what the code should do
        
    Returns:
        Generated code as a string
    """
    config = get_config()
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        api_key=config.openai_api_key
    )
    
    prompt = f"""Generate clean, well-documented {language} code for the following task:

{description}

Requirements:
- Include proper error handling
- Add docstrings/comments
- Follow best practices for {language}
- Make the code production-ready

Return only the code, no explanations."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def validate_python_syntax(code: str) -> Dict[str, Any]:
    """Validate Python code syntax and report any errors.
    
    Args:
        code: Python code to validate
        
    Returns:
        Dictionary with validation results
    """
    try:
        ast.parse(code)
        return {
            "valid": True,
            "errors": [],
            "message": "Code syntax is valid"
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "errors": [{
                "type": "SyntaxError",
                "message": str(e),
                "line": e.lineno,
                "column": e.offset
            }],
            "message": f"Syntax error at line {e.lineno}: {e.msg}"
        }
    except Exception as e:
        return {
            "valid": False,
            "errors": [{
                "type": type(e).__name__,
                "message": str(e)
            }],
            "message": f"Validation error: {str(e)}"
        }


@tool
def explain_code(code: str, language: str = "python") -> str:
    """Explain what a piece of code does.
    
    Args:
        code: Code to explain
        language: Programming language of the code
        
    Returns:
        Explanation of the code
    """
    config = get_config()
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        api_key=config.openai_api_key
    )
    
    prompt = f"""Explain the following {language} code in clear, simple terms:

```{language}
{code}
```

Provide:
1. What the code does overall
2. Key components and their purposes
3. Any important logic or algorithms used
4. Potential improvements or issues"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def generate_tests(code: str, language: str = "python") -> str:
    """Generate unit tests for the given code.
    
    Args:
        code: Code to generate tests for
        language: Programming language
        
    Returns:
        Generated test code
    """
    config = get_config()
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        api_key=config.openai_api_key
    )
    
    if language.lower() == "python":
        framework = "pytest"
    elif language.lower() == "javascript":
        framework = "jest"
    else:
        framework = "appropriate testing framework"
    
    prompt = f"""Generate comprehensive unit tests for this {language} code using {framework}:

```{language}
{code}
```

Requirements:
- Test all major functions/methods
- Include edge cases and error conditions
- Use proper assertions
- Follow {framework} best practices
- Include setup/teardown if needed

Return only the test code."""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def debug_code(code: str, error_message: str) -> str:
    """Debug code based on an error message and suggest fixes.
    
    Args:
        code: Code that has the error
        error_message: Error message or description
        
    Returns:
        Debugging analysis and suggested fixes
    """
    config = get_config()
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        api_key=config.openai_api_key
    )
    
    prompt = f"""Debug this code that's producing an error:

**Code:**
```python
{code}
```

**Error:**
{error_message}

Please provide:
1. Root cause analysis of the error
2. Step-by-step debugging approach
3. Fixed version of the code
4. Prevention tips for similar issues"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


@tool
def run_python_code(code: str) -> Dict[str, Any]:
    """Execute Python code and return the result.
    
    Args:
        code: Python code to execute
        
    Returns:
        Dictionary with execution results
    """
    try:
        # Create a restricted execution environment
        exec_globals = {
            "__builtins__": {
                "print": print,
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "list": list,
                "dict": dict,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "max": max,
                "min": min,
                "sum": sum,
                "abs": abs,
                "round": round,
            }
        }
        
        # Capture output
        import io
        import contextlib
        
        output_buffer = io.StringIO()
        
        with contextlib.redirect_stdout(output_buffer):
            exec(code, exec_globals)
        
        output = output_buffer.getvalue()
        
        return {
            "success": True,
            "output": output,
            "error": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "output": "",
            "error": {
                "type": type(e).__name__,
                "message": str(e)
            }
        }


class CodeAgent:
    """Agent specialized in code-related tasks."""
    
    def __init__(self):
        """Initialize the Code Agent."""
        self.name = "Code Agent"
        self.description = "I handle code generation, debugging, testing, and explanation tasks."
        self.tools = [
            generate_code,
            validate_python_syntax,
            explain_code,
            generate_tests,
            debug_code,
            run_python_code,
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
- generate_code: Generate code in various programming languages
- validate_python_syntax: Check Python code for syntax errors
- explain_code: Explain what code does in simple terms
- generate_tests: Create unit tests for code
- debug_code: Debug code and suggest fixes
- run_python_code: Execute Python code safely

When handling requests:
1. Analyze what the user needs (generation, debugging, testing, etc.)
2. Use the appropriate tools to complete the task
3. Provide clear, helpful responses
4. Always prioritize code quality and best practices

Only handle tasks related to code. If asked about non-code topics, politely redirect to the supervisor."""
    
    def can_handle(self, message: str) -> bool:
        """Determine if this agent can handle the given message."""
        code_keywords = [
            "code", "function", "program", "script", "debug", "test", "syntax",
            "python", "javascript", "sql", "java", "c++", "programming",
            "algorithm", "class", "method", "variable", "loop", "if", "else",
            "import", "library", "framework", "api", "bug", "error", "exception"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in code_keywords)