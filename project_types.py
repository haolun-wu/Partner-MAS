from dataclasses import dataclass


@dataclass
class llm_token_usage:
    input: int
    output: int
    reasoning: int
    cached: int
    total: int

@dataclass
class llm_response_format:
    text: str
    usage: llm_token_usage
    summary: str
    error: str