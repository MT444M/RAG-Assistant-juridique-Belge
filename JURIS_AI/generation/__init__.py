from .generator import generate_text, generate_legal_response
from .prompts import create_generation_prompt, create_hyde_prompt, create_multi_query_prompt

__all__ = ["generate_text",
        "generate_legal_response", 
        "create_generation_prompt", 
        "create_hyde_prompt", 
        "create_multi_query_prompt"]