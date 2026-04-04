"""
AI Provider — Thin adapter for Claude and OpenAI APIs.

Used by the Pack Builder wizard to generate pack structures and code.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Model name mapping: frontend selector value → API model ID
MODEL_MAP = {
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-opus": "claude-opus-4-6",
    "gpt-4": "gpt-4",
    "gpt-4o": "gpt-4o",
}


class AIProviderError(Exception):
    """Raised when an AI provider call fails."""

    def __init__(self, message: str, provider: str, status_code: int | None = None):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


def generate_completion(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 8192,
) -> str:
    """
    Call Claude or OpenAI to generate a text completion.

    Args:
        model: Frontend model key (e.g. 'claude-sonnet', 'gpt-4o')
        system_prompt: System/instruction message
        user_prompt: User message
        max_tokens: Maximum tokens in the response

    Returns:
        The assistant's text response.

    Raises:
        AIProviderError: On missing keys, API errors, or timeouts.
    """
    api_model = MODEL_MAP.get(model)
    if not api_model:
        raise AIProviderError(
            f"Unknown model '{model}'. Valid options: {list(MODEL_MAP.keys())}",
            provider="unknown",
        )

    if model.startswith("claude"):
        return _call_anthropic(api_model, system_prompt, user_prompt, max_tokens)
    else:
        return _call_openai(api_model, system_prompt, user_prompt, max_tokens)


def _call_anthropic(
    model: str, system_prompt: str, user_prompt: str, max_tokens: int
) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise AIProviderError(
            "ANTHROPIC_API_KEY not configured. Add it to your .env file.",
            provider="anthropic",
        )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=120.0)
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    except anthropic.APIStatusError as e:
        logger.error("Anthropic API error: %s %s", e.status_code, e.message)
        raise AIProviderError(
            f"Claude API error: {e.message}",
            provider="anthropic",
            status_code=e.status_code,
        )
    except anthropic.APITimeoutError:
        logger.error("Anthropic API timeout")
        raise AIProviderError(
            "Claude API timed out after 120 seconds. Try a simpler pack or a faster model.",
            provider="anthropic",
        )
    except Exception as e:
        logger.error("Anthropic unexpected error: %s", e)
        raise AIProviderError(
            f"Claude API error: {e}",
            provider="anthropic",
        )


def _call_openai(
    model: str, system_prompt: str, user_prompt: str, max_tokens: int
) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise AIProviderError(
            "OPENAI_API_KEY not configured. Add it to your .env file.",
            provider="openai",
        )

    try:
        import openai

        client = openai.OpenAI(api_key=api_key, timeout=120.0)
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content

    except openai.APIStatusError as e:
        logger.error("OpenAI API error: %s %s", e.status_code, e.message)
        raise AIProviderError(
            f"OpenAI API error: {e.message}",
            provider="openai",
            status_code=e.status_code,
        )
    except openai.APITimeoutError:
        logger.error("OpenAI API timeout")
        raise AIProviderError(
            "OpenAI API timed out after 120 seconds. Try a simpler pack or a faster model.",
            provider="openai",
        )
    except Exception as e:
        logger.error("OpenAI unexpected error: %s", e)
        raise AIProviderError(
            f"OpenAI API error: {e}",
            provider="openai",
        )
