"""File Agent - Handles file operations and system tasks."""

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from supervisor_agent.utils.config import get_config


@tool
def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    """Read contents of a file.
    
    Args:
        file_path: Path to the file to read
        encoding: File encoding (default: utf-8)
        
    Returns:
        Dictionary with file contents and metadata
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}
        
        if not path.is_file():
            return {"error": f"Path is not a file: {file_path}"}
        
        with open(path, 'r', encoding=encoding) as f:
            content = f.read()
        
        stat = path.stat()
        
        return {
            "content": content,
            "file_path": str(path.absolute()),
            "size_bytes": stat.st_size,
            "line_count": len(content.split('\n')),
            "encoding": encoding,
            "exists": True
        }
        
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}


@tool
def write_file(file_path: str, content: str, encoding: str = "utf-8", create_dirs: bool = True) -> Dict[str, Any]:
    """Write content to a file.
    
    Args:
        file_path: Path where to write the file
        content: Content to write
        encoding: File encoding (default: utf-8)
        create_dirs: Create parent directories if they don't exist
        
    Returns:
        Dictionary with operation results
    """
    try:
        path = Path(file_path)
        
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding=encoding) as f:
            f.write(content)
        
        stat = path.stat()
        
        return {
            "success": True,
            "file_path": str(path.absolute()),
            "size_bytes": stat.st_size,
            "line_count": len(content.split('\n')),
            "encoding": encoding
        }
        
    except Exception as e:
        return {"error": f"Failed to write file: {str(e)}"}


@tool
def list_directory(directory_path: str, include_hidden: bool = False, file_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """List contents of a directory.
    
    Args:
        directory_path: Path to the directory
        include_hidden: Include hidden files (starting with .)
        file_types: Filter by file extensions (e.g., ['.py', '.txt'])
        
    Returns:
        Dictionary with directory contents
    """
    try:
        path = Path(directory_path)
        if not path.exists():
            return {"error": f"Directory not found: {directory_path}"}
        
        if not path.is_dir():
            return {"error": f"Path is not a directory: {directory_path}"}
        
        files = []
        directories = []
        
        for item in path.iterdir():
            if not include_hidden and item.name.startswith('.'):
                continue
                
            if item.is_file():
                if file_types and item.suffix.lower() not in [ext.lower() for ext in file_types]:
                    continue
                    
                stat = item.stat()
                files.append({
                    "name": item.name,
                    "path": str(item.absolute()),
                    "size_bytes": stat.st_size,
                    "extension": item.suffix,
                    "modified": stat.st_mtime
                })
            elif item.is_dir():
                directories.append({
                    "name": item.name,
                    "path": str(item.absolute())
                })
        
        return {
            "directory_path": str(path.absolute()),
            "files": sorted(files, key=lambda x: x["name"]),
            "directories": sorted(directories, key=lambda x: x["name"]),
            "total_files": len(files),
            "total_directories": len(directories)
        }
        
    except Exception as e:
        return {"error": f"Failed to list directory: {str(e)}"}


@tool
def create_directory(directory_path: str, parents: bool = True) -> Dict[str, Any]:
    """Create a directory.
    
    Args:
        directory_path: Path of the directory to create
        parents: Create parent directories if they don't exist
        
    Returns:
        Dictionary with operation results
    """
    try:
        path = Path(directory_path)
        path.mkdir(parents=parents, exist_ok=True)
        
        return {
            "success": True,
            "directory_path": str(path.absolute()),
            "created": True
        }
        
    except Exception as e:
        return {"error": f"Failed to create directory: {str(e)}"}


@tool
def delete_file_or_directory(path: str, recursive: bool = False) -> Dict[str, Any]:
    """Delete a file or directory.
    
    Args:
        path: Path to delete
        recursive: If True, delete directories recursively
        
    Returns:
        Dictionary with operation results
    """
    try:
        target_path = Path(path)
        if not target_path.exists():
            return {"error": f"Path not found: {path}"}
        
        if target_path.is_file():
            target_path.unlink()
            return {
                "success": True,
                "deleted_path": str(target_path.absolute()),
                "type": "file"
            }
        elif target_path.is_dir():
            if recursive:
                shutil.rmtree(target_path)
                return {
                    "success": True,
                    "deleted_path": str(target_path.absolute()),
                    "type": "directory",
                    "recursive": True
                }
            else:
                target_path.rmdir()  # Only works if directory is empty
                return {
                    "success": True,
                    "deleted_path": str(target_path.absolute()),
                    "type": "directory",
                    "recursive": False
                }
        
    except Exception as e:
        return {"error": f"Failed to delete: {str(e)}"}


@tool
def copy_file_or_directory(source: str, destination: str) -> Dict[str, Any]:
    """Copy a file or directory.
    
    Args:
        source: Source path
        destination: Destination path
        
    Returns:
        Dictionary with operation results
    """
    try:
        src_path = Path(source)
        dst_path = Path(destination)
        
        if not src_path.exists():
            return {"error": f"Source not found: {source}"}
        
        if src_path.is_file():
            # Create destination directory if needed
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            return {
                "success": True,
                "source": str(src_path.absolute()),
                "destination": str(dst_path.absolute()),
                "type": "file"
            }
        elif src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            return {
                "success": True,
                "source": str(src_path.absolute()),
                "destination": str(dst_path.absolute()),
                "type": "directory"
            }
        
    except Exception as e:
        return {"error": f"Failed to copy: {str(e)}"}


@tool
def search_in_files(directory: str, pattern: str, file_types: Optional[List[str]] = None, case_sensitive: bool = False) -> Dict[str, Any]:
    """Search for text pattern in files within a directory.
    
    Args:
        directory: Directory to search in
        pattern: Text pattern to search for
        file_types: File extensions to search in (e.g., ['.py', '.txt'])
        case_sensitive: Whether search should be case sensitive
        
    Returns:
        Dictionary with search results
    """
    try:
        import re
        
        path = Path(directory)
        if not path.exists() or not path.is_dir():
            return {"error": f"Directory not found: {directory}"}
        
        search_pattern = pattern if case_sensitive else pattern.lower()
        results = []
        
        for file_path in path.rglob('*'):
            if not file_path.is_file():
                continue
                
            if file_types and file_path.suffix.lower() not in [ext.lower() for ext in file_types]:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    search_content = content if case_sensitive else content.lower()
                    
                    if search_pattern in search_content:
                        # Find line numbers where pattern occurs
                        lines = content.split('\n')
                        matching_lines = []
                        
                        for i, line in enumerate(lines, 1):
                            line_search = line if case_sensitive else line.lower()
                            if search_pattern in line_search:
                                matching_lines.append({
                                    "line_number": i,
                                    "content": line.strip()
                                })
                        
                        results.append({
                            "file_path": str(file_path.absolute()),
                            "matches": len(matching_lines),
                            "matching_lines": matching_lines
                        })
                        
            except Exception:
                # Skip files that can't be read
                continue
        
        return {
            "pattern": pattern,
            "directory": str(path.absolute()),
            "case_sensitive": case_sensitive,
            "file_types": file_types,
            "total_files_with_matches": len(results),
            "results": results
        }
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


@tool
def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get detailed information about a file or directory.
    
    Args:
        file_path: Path to analyze
        
    Returns:
        Dictionary with file information
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"Path not found: {file_path}"}
        
        stat = path.stat()
        
        info = {
            "path": str(path.absolute()),
            "name": path.name,
            "parent": str(path.parent.absolute()),
            "size_bytes": stat.st_size,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "permissions": oct(stat.st_mode)[-3:]
        }
        
        if path.is_file():
            info.update({
                "extension": path.suffix,
                "stem": path.stem,
                "size_mb": round(stat.st_size / (1024 * 1024), 2)
            })
            
            # Try to determine file type
            if path.suffix.lower() in ['.txt', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
                info["file_type"] = "text"
            elif path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                info["file_type"] = "image"
            elif path.suffix.lower() in ['.pdf', '.doc', '.docx']:
                info["file_type"] = "document"
            else:
                info["file_type"] = "binary"
        
        return info
        
    except Exception as e:
        return {"error": f"Failed to get file info: {str(e)}"}


@tool
def analyze_file_content(file_path: str) -> str:
    """Analyze file content and provide insights using LLM.
    
    Args:
        file_path: Path to the file to analyze
        
    Returns:
        Analysis of the file content
    """
    config = get_config()
    llm = ChatOpenAI(
        model=config.openai_model,
        temperature=config.temperature,
        api_key=config.openai_api_key
    )
    
    # Read the file first
    file_result = read_file(file_path)
    if "error" in file_result:
        return f"Error reading file: {file_result['error']}"
    
    content = file_result["content"]
    file_info = get_file_info(file_path)
    
    prompt = f"""Analyze this file and provide insights:

**File Information:**
- Path: {file_path}
- Size: {file_info.get('size_bytes', 0)} bytes
- Type: {file_info.get('file_type', 'unknown')}

**Content:**
```
{content[:2000]}  # Limit content for analysis
{'...(truncated)' if len(content) > 2000 else ''}
```

Please provide:
1. What type of file this is and its purpose
2. Key content analysis (structure, format, etc.)
3. Any issues or recommendations
4. Summary of what the file contains"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


class FileAgent:
    """Agent specialized in file operations and system tasks."""
    
    def __init__(self):
        """Initialize the File Agent."""
        self.name = "File Agent"
        self.description = "I handle file operations, directory management, and system tasks."
        self.tools = [
            read_file,
            write_file,
            list_directory,
            create_directory,
            delete_file_or_directory,
            copy_file_or_directory,
            search_in_files,
            get_file_info,
            analyze_file_content,
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
- read_file: Read contents of any file
- write_file: Write content to files
- list_directory: List directory contents
- create_directory: Create new directories
- delete_file_or_directory: Delete files or directories
- copy_file_or_directory: Copy files or directories
- search_in_files: Search for text patterns in files
- get_file_info: Get detailed file/directory information
- analyze_file_content: Analyze file content with AI insights

When handling requests:
1. Always verify file/directory paths exist before operations
2. Use appropriate error handling and provide clear feedback
3. Be cautious with delete operations - confirm before executing
4. Create necessary parent directories when writing files
5. Provide detailed results and status information

Only handle tasks related to file system operations. If asked about non-file topics, politely redirect to the supervisor."""
    
    def can_handle(self, message: str) -> bool:
        """Determine if this agent can handle the given message."""
        file_keywords = [
            "file", "directory", "folder", "read", "write", "create", "delete",
            "copy", "move", "list", "search", "find", "path", "save", "load",
            "text", "document", "csv", "json", "config", "log", "backup",
            "organize", "manage", "browse", "explore", "analyze", "content"
        ]
        
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in file_keywords)