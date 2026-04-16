import os

from dotenv import load_dotenv
from openai import OpenAI


class OpenAIClient:
    def __init__(self, env_path: str = ".env") -> None:
        load_dotenv(env_path)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found. Add it to your .env file.")
        self.client = OpenAI(api_key=api_key)

