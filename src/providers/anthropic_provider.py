import base64

import anthropic

_client = None


def call(model_id, image_bytes, prompt, mime_type):
    global _client
    _client = _client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    response = _client.messages.create(
        model=model_id,
        max_tokens=800,
        # No temperature: Claude API version 2026-10-21 (anthropic SDK >= 2.0) rejects
        # sampling parameters, so Claude runs at its default sampling. See docs/FINDINGS.md.
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": mime_type,
                    "data": base64.b64encode(image_bytes).decode(),
                }},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = "".join(block.text for block in response.content if block.type == "text")
    return text, response.usage.input_tokens, response.usage.output_tokens
