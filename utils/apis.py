import openai
import anthropic
import os
import time
from dotenv import load_dotenv

load_dotenv("../secrets.env")

# TODO: add anthropic models too
openai.api_key = os.environ.get("OPENAI_API_KEY")
anthropic.api_key = os.environ.get("ANTHROPIC_API_KEY")

openai_client = openai.Client()
anthropic_client = anthropic.Anthropic()


class LLM_API:
    def __init__(self, api_func, name=None):
        self.api_func = api_func
        self.api_name = name

    def __call__(self, messages, model_override=None):
        return self.api_func(messages, model_override)

    def name(self):
        return self.api_name


class OpenAI_API(LLM_API):
    def __init__(self):
        super().__init__(api_func=ask_gpt, name="OpenAI")


class Anthropic_API(LLM_API):
    def __init__(self):
        super().__init__(api_func=ask_anthropic, name="Anthropic")


def execute_with_exp_backoff(func, *args, **kwargs):
    for retry in range(2):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Retrying in {2**retry}... ", e)
            time.sleep(2**retry)
            raise e


def ask_gpt(messages, model_override=None):
    model = model_override or "gpt-4o-mini"
    try:
        response = openai_client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def ask_anthropic(messages, model_override=None):
    model = model_override or "claude-3-7-sonnet-20250219"
    try:
        response = execute_with_exp_backoff(
            anthropic_client.messages.create,
            model=model,
            max_tokens=1024,
            messages=messages,
        )
        return response.content[0].text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
