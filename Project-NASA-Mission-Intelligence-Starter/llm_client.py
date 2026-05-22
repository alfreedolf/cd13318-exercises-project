"""Module responsible for all the LLM client related implementations."""

import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict
from openai import OpenAI



def load_system_prompt(filename: str = "system_prompt.txt") -> str:
    prompt_path = Path(__file__).parent / filename
    if not prompt_path.exists():
        raise FileNotFoundError(
            f"System prompt file not found at: {prompt_path}. "
            "Ensure 'system_prompt.txt' exists in the project directory."
        )
    return prompt_path.read_text(encoding="utf-8")


def generate_response(
    openai_key: str, user_message: str, context: str, conversation_history: List[Dict], model: str = "gpt-3.5-turbo"
) -> str | None:
    """Generate response using OpenAI with context"""

    # TODO: Define system prompt
    system_prompt = load_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history,
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_message}"},
    ]
    # TODO: Set context in messages
    messages.append({"role": "user", "content": context})
    # TODO: Add chat history
    messages.extend(conversation_history)
    # TODO: Creaet OpenAI Client
    load_dotenv()
    openai_api_key = os.environ["OPENAI_API_KEY"]
    vocareum_base_url = os.environ["VOCAREUM_BASE_URL"]
    openai_client = OpenAI(api_key=openai_api_key, base_url=vocareum_base_url)
    # TODO: Send request to OpenAI
    response = openai_client.chat.completions.create(model=model, messages=messages)
    # Get assistant's response
    assistant_message = response.choices[0].message.content
    
    # TODO: Return response
    return assistant_message
