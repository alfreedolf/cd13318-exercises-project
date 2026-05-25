import sys
import os
import types
from dotenv import load_dotenv
from pathlib import Path

import pytest

# retrieving key from .env file
load_dotenv()
opena_api_key = os.environ["OPENAI_API_VOCAREUM_KEY"]

# Ensure the module path is available when pytest runs from the repository root.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Mock the openai package so the module imports successfully in environments
# where OpenAI is not installed.
fake_openai = types.ModuleType("openai")
fake_openai.OpenAI = lambda api_key, base_url: None
sys.modules["openai"] = fake_openai

import llm_client


def test_load_system_prompt_reads_existing_file(tmp_path, monkeypatch):
    prompt_text = "You are NASA mission control."
    fake_module_path = tmp_path / "module_dir"
    fake_module_path.mkdir()
    prompt_file = fake_module_path / "system_prompt.txt"
    prompt_file.write_text(prompt_text, encoding="utf-8")

    monkeypatch.setattr(llm_client, "__file__", str(fake_module_path / "llm_client.py"))

    assert llm_client.load_system_prompt() == prompt_text


def test_load_system_prompt_raises_when_file_missing(tmp_path, monkeypatch):
    fake_module_path = tmp_path / "module_dir"
    fake_module_path.mkdir()
    monkeypatch.setattr(llm_client, "__file__", str(fake_module_path / "llm_client.py"))

    with pytest.raises(FileNotFoundError) as exc_info:
        llm_client.load_system_prompt()

    assert "System prompt file not found" in str(exc_info.value)


class DummyCompletionResult:
    def __init__(self, content):
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class DummyCompletions:
    def __init__(self, response):
        self.response = response
        self.last_model = None
        self.last_messages = None

    def create(self, model, messages):
        self.last_model = model
        self.last_messages = messages
        return self.response


class DummyChat:
    def __init__(self, response):
        self.completions = DummyCompletions(response)


class DummyOpenAI:
    def __init__(self, api_key, response=None):
        self.api_key = api_key
        self.chat = DummyChat(response)


def test_generate_response_calls_openai_and_returns_assistant_message():
    # system_prompt = "System prompt for NASA."
    # monkeypatch.setattr(llm_client, "load_system_prompt", lambda: system_prompt)

    assistant_text = "Roger that, Commander."
    # response = DummyCompletionResult(assistant_text)
    captured = {}

    # def fake_openai_factory(api_key):
    #     client = DummyOpenAI(api_key, response=response)
    #     captured["client"] = client
    #     return client

    # monkeypatch.setattr(llm_client, "OpenAI", fake_openai_factory)

    user_message = "What is our next step?"
    context = "We are currently in lunar orbit."
    conversation_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi, how can I help?"},
    ]

    result = llm_client.generate_response(
        openai_key=opena_api_key,
        user_message=user_message,
        context=context,
        conversation_history=conversation_history
    )

    assert result == assistant_text
    assert captured["client"].chat.completions.last_messages[0]["role"] == "system"
    assert captured["client"].chat.completions.last_messages[-1]["content"] == (
        f"Context:\n{context}\n\nQuestion: {user_message}"
    )
