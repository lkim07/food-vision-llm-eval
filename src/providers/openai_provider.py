import base64

from openai import OpenAI

_client = None


def call(model_id, image_bytes, prompt, mime_type):
    global _client
    _client = _client or OpenAI()  # reads OPENAI_API_KEY
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"
    response = _client.chat.completions.create(
        model=model_id,
        temperature=0,
        max_tokens=800,
        response_format={"type": "json_object"},
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }],
    )
    usage = response.usage
    return response.choices[0].message.content, usage.prompt_tokens, usage.completion_tokens
