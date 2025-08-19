"""
Unified LLM client supporting both Claude and Gemini APIs for RAG system.
"""

import time
import random
import re
from typing import Optional, Dict, Any
from src.config import GEMINI_API_KEY, CLAUDE_API_KEY, PREFERRED_LLM, GEMINI_MODEL_NAME, CLAUDE_MODEL_NAME


class LLMClient:
    """Unified client for multiple LLM providers with failover support."""
    
    def __init__(self, preferred_provider: str = None):
        self.preferred_provider = preferred_provider or PREFERRED_LLM
        self.gemini_client = None
        self.claude_client = None
        
        # Initialize available clients
        self._init_gemini()
        self._init_claude()
        
        # Determine which client to use
        if self.preferred_provider == "claude" and self.claude_client:
            self.primary_client = "claude"
            self.fallback_client = "gemini" if self.gemini_client else None
        else:
            self.primary_client = "gemini" if self.gemini_client else None
            self.fallback_client = "claude" if self.claude_client else None
        
        print(f"LLM Client initialized - Primary: {self.primary_client}, Fallback: {self.fallback_client}")
    
    def _init_gemini(self):
        """Initialize Gemini client if API key is available."""
        if not GEMINI_API_KEY:
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_client = genai.GenerativeModel(GEMINI_MODEL_NAME)
            print("Gemini client initialized")
        except ImportError:
            print("google-generativeai not installed, Gemini unavailable")
        except Exception as e:
            print(f"Failed to initialize Gemini client: {e}")
    
    def _init_claude(self):
        """Initialize Claude client if API key is available."""
        if not CLAUDE_API_KEY:
            return
        
        try:
            import anthropic
            self.claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
            print("Claude client initialized")
        except ImportError:
            print("anthropic not installed, Claude unavailable")
            print("Install with: pip install anthropic")
        except Exception as e:
            print(f"Failed to initialize Claude client: {e}")
    
    def generate_content(self, prompt: str, max_retries: int = 3) -> str:
        """
        Generate content using the primary LLM with fallback support.
        
        Args:
            prompt: Input prompt
            max_retries: Maximum retry attempts for rate limiting
            
        Returns:
            Generated text content
        """
        # Try primary client first
        if self.primary_client:
            try:
                return self._generate_with_provider(prompt, self.primary_client, max_retries)
            except Exception as e:
                print(f"Primary provider {self.primary_client} failed: {e}")
                
        # Try fallback client
        if self.fallback_client:
            try:
                print(f"Trying fallback provider: {self.fallback_client}")
                return self._generate_with_provider(prompt, self.fallback_client, max_retries)
            except Exception as e:
                print(f"Fallback provider {self.fallback_client} failed: {e}")
        
        return "Error: No available LLM providers could generate content"
    
    def _generate_with_provider(self, prompt: str, provider: str, max_retries: int) -> str:
        """Generate content with a specific provider."""
        for attempt in range(max_retries):
            try:
                if provider == "claude":
                    return self._generate_claude(prompt)
                elif provider == "gemini":
                    return self._generate_gemini(prompt)
                else:
                    raise ValueError(f"Unknown provider: {provider}")
                    
            except Exception as e:
                if self._is_rate_limit_error(e) and attempt < max_retries - 1:
                    wait_time = (3 ** attempt) + random.uniform(1, 3)
                    print(f"Rate limit hit ({provider}), waiting {wait_time:.1f}s before retry {attempt+1}/{max_retries}")
                    time.sleep(wait_time)
                    continue
                else:
                    raise e
        
        return f"Error: {provider} failed after {max_retries} attempts"
    
    def _generate_claude(self, prompt: str) -> str:
        """Generate content using Claude API with model fallback."""
        if not self.claude_client:
            raise Exception("Claude client not initialized")
        
        # Try primary model first, fallback to Haiku if it fails
        models_to_try = [CLAUDE_MODEL_NAME, "claude-3-haiku-20240307"]
        
        for model in models_to_try:
            try:
                response = self.claude_client.messages.create(
                    model=model,
                    max_tokens=2048,
                    temperature=0.1,  # Low temperature for consistent legal analysis
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                
                if response.content:
                    return response.content[0].text
                
            except Exception as e:
                if "not_found" in str(e).lower() and model != models_to_try[-1]:
                    print(f"Model {model} not found, trying fallback...")
                    continue
                else:
                    raise e
                    
        raise Exception("All Claude models failed")
    
    def _generate_gemini(self, prompt: str) -> str:
        """Generate content using Gemini API."""
        if not self.gemini_client:
            raise Exception("Gemini client not initialized")
            
        response = self.gemini_client.generate_content(prompt)
        
        # Handle Gemini response format
        if response.parts:
            return "".join(part.text for part in response.parts if hasattr(part, 'text'))
        elif hasattr(response, 'text') and response.text:
            return response.text
        else:
            raise Exception("Empty response from Gemini")
    
    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is due to rate limiting."""
        error_str = str(error).lower()
        rate_limit_indicators = [
            "429", "rate limit", "quota", "too many requests",
            "rate_limit_exceeded", "throttle"
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)
    
    def extract_score(self, text: str) -> Optional[float]:
        """Extract numerical score from LLM response."""
        # Look for decimal numbers between 0 and 1
        matches = re.findall(r'\b([0-1]?\.\d+)\b', text)
        for match in matches:
            try:
                score = float(match)
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
        
        # Look for simple decimals
        match = re.search(r'\b(\d\.\d+)\b', text)
        if match:
            try:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        return None
    
    def is_available(self) -> bool:
        """Check if any LLM provider is available."""
        return bool(self.primary_client or self.fallback_client)


# Global instance
_llm_client = None

def get_llm_client() -> LLMClient:
    """Get the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


if __name__ == "__main__":
    # Test the client
    client = LLMClient()
    
    test_prompt = "What is the capital of France? Answer briefly."
    
    if client.is_available():
        print("Testing LLM client...")
        result = client.generate_content(test_prompt)
        print(f"Result: {result}")
    else:
        print("No LLM providers available for testing")