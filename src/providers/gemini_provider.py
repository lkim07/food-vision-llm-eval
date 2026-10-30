from google import genai
from google.genai import types

_client = None


def call(model_id, image_bytes, prompt, mime_type):
    global _client
    _client = _client or genai.Client()  # reads GEMINI_API_KEY
    response = _client.models.generate_content(
        model=model_id,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), prompt],
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=800,
            response_mime_type="application/json",
        ),
    )
    usage = response.usage_metadata
    return response.text, usage.prompt_token_count, usage.candidates_token_count
