"""
orchestra_sdk.constants
========================
Shared constants used across multiple modules.
"""

# Conservative approximation: 1 token ≈ 4 characters.
# Used by context.py and tools/file_tools.py for token budget enforcement.
# If switching to a real tokenizer (e.g. tiktoken), update this value or
# replace the usages with a proper token-counting function.
CHARS_PER_TOKEN: int = 4
