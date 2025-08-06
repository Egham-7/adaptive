"""
HuggingFace Research Tool

Researches model metadata from HuggingFace Hub model cards and repository information.
"""

import asyncio
import logging
from typing import Optional
import httpx
from bs4 import BeautifulSoup
import re

from .base_research_tool import BaseResearchTool, ModelMetadata

logger = logging.getLogger(__name__)


class HuggingFaceResearchTool(BaseResearchTool):
    """Research tool for HuggingFace Hub model information"""
    
    def __init__(self):
        self.base_url = "https://huggingface.co"
        self.api_url = "https://huggingface.co/api"
        
    async def research_model(
        self, 
        provider: str, 
        model_name: str, 
        **kwargs
    ) -> Optional[ModelMetadata]:
        """
        Research model metadata from HuggingFace Hub
        
        Args:
            provider: Provider name (e.g., "ANTHROPIC", "OPENAI")
            model_name: Model name (e.g., "claude-3-sonnet")
            **kwargs: Additional research parameters
            
        Returns:
            ModelMetadata object with discovered information or None
        """
        try:
            logger.info(f"🔍 Researching {provider}/{model_name} on HuggingFace")
            
            # Try different model name variations for HuggingFace search
            search_terms = self._generate_search_terms(provider, model_name)
            
            for search_term in search_terms:
                logger.debug(f"Searching HF for: {search_term}")
                metadata = await self._search_huggingface(search_term)
                if metadata and self._has_useful_metadata(metadata):
                    logger.info(f"✅ Found HF metadata for {model_name}")
                    return metadata
                    
            logger.warning(f"❌ No useful HF metadata found for {model_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error researching {model_name} on HuggingFace: {e}")
            return None
            
    def _generate_search_terms(self, provider: str, model_name: str) -> list[str]:
        """Generate search terms for HuggingFace based on model name"""
        terms = []
        
        # Clean model name
        clean_name = model_name.replace("-", " ").replace("_", " ")
        
        # Add direct model name
        terms.append(model_name)
        terms.append(clean_name)
        
        # Add provider + model combinations
        provider_lower = provider.lower()
        if provider_lower == "anthropic":
            terms.extend([
                f"anthropic {model_name}",
                f"claude {clean_name}",
                clean_name.replace("claude", "").strip()
            ])
        elif provider_lower == "openai":
            terms.extend([
                f"openai {model_name}",
                f"gpt {clean_name}",
                clean_name.replace("gpt", "").strip()
            ])
        elif provider_lower == "google":
            terms.extend([
                f"google {model_name}",
                f"gemini {clean_name}",
                clean_name.replace("gemini", "").strip()
            ])
        
        # Remove empty terms
        return [term for term in terms if term.strip()]
        
    async def _search_huggingface(self, search_term: str) -> Optional[ModelMetadata]:
        """Search HuggingFace Hub for model information"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Search for models
                search_url = f"{self.api_url}/models"
                params = {
                    "search": search_term,
                    "sort": "downloads",
                    "direction": -1,
                    "limit": 5
                }
                
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                
                models = response.json()
                
                for model in models:
                    model_id = model.get("id", "")
                    if not model_id:
                        continue
                        
                    # Get detailed model info
                    model_metadata = await self._get_model_details(client, model_id)
                    if model_metadata:
                        return model_metadata
                        
                return None
                
        except Exception as e:
            logger.error(f"Error searching HuggingFace: {e}")
            return None
            
    async def _get_model_details(self, client: httpx.AsyncClient, model_id: str) -> Optional[ModelMetadata]:
        """Get detailed information about a specific model"""
        try:
            # Get model card HTML
            model_url = f"{self.base_url}/{model_id}"
            response = await client.get(model_url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract model information
            metadata = ModelMetadata()
            
            # Description from README or model card
            readme = soup.find("div", {"id": "readme"})
            if readme:
                # Extract description from first paragraph or header
                description = self._extract_description(readme)
                if description:
                    metadata.description = description
                    
            # Look for model details in the page
            metadata = self._extract_model_specs(soup, metadata)
            
            # Get model API info if available
            api_metadata = await self._get_model_api_info(client, model_id)
            if api_metadata:
                metadata = metadata.merge(api_metadata)
                
            return metadata if self._has_useful_metadata(metadata) else None
            
        except Exception as e:
            logger.error(f"Error getting model details for {model_id}: {e}")
            return None
            
    def _extract_description(self, readme_div) -> Optional[str]:
        """Extract a clean description from README content"""
        try:
            # Look for the first meaningful paragraph
            paragraphs = readme_div.find_all("p")
            for p in paragraphs:
                text = p.get_text().strip()
                if len(text) > 50 and not text.lower().startswith("this is"):
                    # Clean up the text
                    text = re.sub(r'\s+', ' ', text)
                    return text[:500] + "..." if len(text) > 500 else text
                    
            # Try headings if no good paragraph found
            headings = readme_div.find_all(["h1", "h2", "h3"])
            for h in headings[1:3]:  # Skip the first heading (usually model name)
                text = h.get_text().strip()
                if len(text) > 20:
                    return text
                    
            return None
            
        except Exception as e:
            logger.error(f"Error extracting description: {e}")
            return None
            
    def _extract_model_specs(self, soup: BeautifulSoup, metadata: ModelMetadata) -> ModelMetadata:
        """Extract model specifications from the page"""
        try:
            # Look for common patterns in model specifications
            text = soup.get_text().lower()
            
            # Context window detection
            context_patterns = [
                r'context(?:\s+window)?(?:\s+size)?:?\s*(\d+(?:k|m)?)',
                r'(\d+(?:k|m)?)\s*context(?:\s+window)?',
                r'max(?:imum)?\s+context:?\s*(\d+(?:k|m)?)',
                r'context(?:\s+length)?:?\s*(\d+(?:k|m)?)'
            ]
            
            for pattern in context_patterns:
                match = re.search(pattern, text)
                if match:
                    tokens = self._parse_token_count(match.group(1))
                    if tokens:
                        metadata.max_context_tokens = tokens
                        break
                        
            # Output tokens detection
            output_patterns = [
                r'output(?:\s+tokens?)?:?\s*(\d+(?:k|m)?)',
                r'max(?:imum)?\s+output:?\s*(\d+(?:k|m)?)',
                r'generation(?:\s+length)?:?\s*(\d+(?:k|m)?)'
            ]
            
            for pattern in output_patterns:
                match = re.search(pattern, text)
                if match:
                    tokens = self._parse_token_count(match.group(1))
                    if tokens:
                        metadata.max_output_tokens = tokens
                        break
                        
            # Function calling detection
            if any(term in text for term in [
                'function call', 'tool use', 'api call', 'function', 'tools'
            ]):
                metadata.supports_function_calling = True
                
            # Language support detection
            languages = self._detect_languages(text)
            if languages:
                metadata.languages_supported = languages
                
            # Model size detection
            size_patterns = [
                r'(\d+(?:\.\d+)?)\s*b(?:illion)?\s*param',
                r'(\d+(?:\.\d+)?)\s*m(?:illion)?\s*param',
                r'param(?:eter)?s?:?\s*(\d+(?:\.\d+)?)\s*[bm]'
            ]
            
            for pattern in size_patterns:
                match = re.search(pattern, text)
                if match:
                    size = match.group(1)
                    if 'b' in match.group(0).lower():
                        metadata.model_size_params = f"{size}B"
                    elif 'm' in match.group(0).lower():
                        metadata.model_size_params = f"{size}M"
                    break
                    
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting model specs: {e}")
            return metadata
            
    def _parse_token_count(self, token_str: str) -> Optional[int]:
        """Parse token count from string like '32k' or '1m'"""
        try:
            token_str = token_str.lower().strip()
            
            if token_str.endswith('k'):
                return int(float(token_str[:-1]) * 1000)
            elif token_str.endswith('m'):
                return int(float(token_str[:-1]) * 1000000)
            else:
                return int(token_str)
                
        except (ValueError, TypeError):
            return None
            
    def _detect_languages(self, text: str) -> list[str]:
        """Detect supported languages from text"""
        language_map = {
            'english': 'en',
            'spanish': 'es', 'español': 'es',
            'french': 'fr', 'français': 'fr',
            'german': 'de', 'deutsch': 'de',
            'italian': 'it', 'italiano': 'it',
            'portuguese': 'pt', 'português': 'pt',
            'russian': 'ru', 'русский': 'ru',
            'japanese': 'ja', '日本語': 'ja',
            'chinese': 'zh', '中文': 'zh', 'mandarin': 'zh',
            'korean': 'ko', '한국어': 'ko',
            'arabic': 'ar', 'العربية': 'ar',
            'hindi': 'hi', 'हिन्दी': 'hi'
        }
        
        detected = set()
        text_lower = text.lower()
        
        for lang_name, lang_code in language_map.items():
            if lang_name in text_lower:
                detected.add(lang_code)
                
        # Always include English if multilingual mentioned
        if 'multilingual' in text_lower or len(detected) > 1:
            detected.add('en')
            
        return sorted(list(detected))
        
    async def _get_model_api_info(self, client: httpx.AsyncClient, model_id: str) -> Optional[ModelMetadata]:
        """Get model API information if available"""
        try:
            # Try to get model info from API
            api_url = f"{self.api_url}/models/{model_id}"
            response = await client.get(api_url)
            response.raise_for_status()
            
            model_info = response.json()
            metadata = ModelMetadata()
            
            # Extract any useful API information
            if 'pipeline_tag' in model_info:
                tag = model_info['pipeline_tag']
                if tag == 'text-generation':
                    metadata.task_type = 'Text Generation'
                elif tag == 'conversational':
                    metadata.task_type = 'Conversational'
                    
            return metadata
            
        except Exception as e:
            logger.debug(f"Could not get API info for {model_id}: {e}")
            return None
            
    def _has_useful_metadata(self, metadata: ModelMetadata) -> bool:
        """Check if the metadata contains useful information"""
        return any([
            metadata.description,
            metadata.max_context_tokens,
            metadata.max_output_tokens,
            metadata.supports_function_calling is not None,
            metadata.languages_supported,
            metadata.model_size_params,
            metadata.task_type
        ])