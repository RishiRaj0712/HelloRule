import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "llama-3.1-8b-instant"


class Generator:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found.\n"
            )
        self.client = Groq(api_key=api_key)
        self.model  = GROQ_MODEL
        print(f"[Generator] Ready — using Groq/{GROQ_MODEL}")

    def _is_rate_limit(self, error_msg: str) -> bool:
        keywords = ["429", "rate_limit", "rate limit", "too many requests"]
        return any(k in error_msg.lower() for k in keywords)

    def generate(self, prompt: str) -> str:
        system_msg, user_msg = _split_prompt(prompt)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                error_msg = str(e)

                if self._is_rate_limit(error_msg):
                    wait = 15 * (attempt + 1)   # 15s, 30s, 45s
                    print(f"[Generator] Rate limited — waiting {wait}s (attempt {attempt+1}/{max_attempts})")
                    time.sleep(wait)
                    continue

                else:
                    print(f"[Generator] Error: {error_msg}")
                    return f"An error occurred while generating the response: {error_msg}"

        return "Rate limit reached. Please wait a moment and try again."

    def generate_stream(self, prompt: str):
        system_msg, user_msg = _split_prompt(prompt)

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user",   "content": user_msg},
                    ],
                    temperature=0.3,
                    max_tokens=1024,
                    stream=True,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
                return  # success

            except Exception as e:
                error_msg = str(e)
                if self._is_rate_limit(error_msg) and attempt < max_attempts - 1:
                    wait = 15 * (attempt + 1)
                    print(f"[Generator] Stream rate limited — waiting {wait}s")
                    time.sleep(wait)
                    continue
                yield f"\n\n[Error: {error_msg}]"
                return


def _split_prompt(prompt: str) -> tuple[str, str]:
    """
    Split our assembled prompt into system + user messages.

    Our prompt.py builds:
      SYSTEM_PROMPT
      ====
      CONTEXT FROM THE CONSTITUTION...
      ====
      CURRENT QUESTION: ...
      ====
      ANSWER:

    Groq's chat API needs these as separate roles.
    We treat everything up to the first === as the system message,
    and everything after as the user message.
    """
    separator = "=" * 60
    parts = prompt.split(separator, 1)

    if len(parts) == 2:
        system_msg = parts[0].strip()
        user_msg   = parts[1].strip()
    else:
        # Fallback — put everything in user message
        system_msg = "You are LawBook, a legal assistant for the Constitution of India."
        user_msg   = prompt

    return system_msg, user_msg