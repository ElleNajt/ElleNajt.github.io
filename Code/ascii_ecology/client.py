"""Haiku API wrapper for the primordial soup experiment."""

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-haiku-4-5-20251001"

def call_haiku(input_text: str, max_tokens: int = 256, system: str | None = None,
               prefill: bool = False, user_prompt: str = "Continue:") -> str:
    if prefill:
        prefill_text = input_text.rstrip()
        if not prefill_text:
            prefill_text = "..."
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": prefill_text},
        ]
    else:
        messages = [{"role": "user", "content": input_text}]

    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=0,
        messages=messages,
    )
    if system:
        kwargs["system"] = system

    response = client.messages.create(**kwargs)
    if not response.content or len(response.content) == 0:
        return ""
    block = response.content[0]
    if block.type != "text":
        return ""
    return block.text
