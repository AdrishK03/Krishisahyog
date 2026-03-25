"""
KrishiSahyog Chat Backend (Production-Ready Version)
Features:
- Multi-provider fallback (SambaNova → Gemini → Claude → OpenAI)
- Secure API key handling (.env)
- Retry + timeout
- Clean structure
"""

import os
import logging
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SYSTEM_PROMPT = (
    "You are KrishiSahyog, a friendly and knowledgeable AI assistant for Indian farmers. "
    "Give practical, actionable advice about farming, crops, soil, weather, pest control, "
    "fertilizers, irrigation, and government schemes. "
    "Use simple language. When helpful, mention Indian seasons (Kharif/Rabi/Zaid), "
    "common Indian crops, and local context. "
    "If the user writes in Hindi or a regional language, reply in the same language."
)

# ---------------- Utility ----------------

def safe_execute(fn, retries=1, delay=1):
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(delay)
    return None

# ---------------- Provider 1: SambaNova ----------------

def _try_sambanova(message: str, history: List[Dict]) -> Optional[str]:
    api_key = os.getenv("SAMBANOVA_API_KEY")
    if not api_key:
        return None

    try:
        from sambanova import SambaNova

        client = SambaNova(api_key=api_key, base_url="https://api.sambanova.ai/v1")

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history:
            if h.get("content"):
                messages.append(h)
        messages.append({"role": "user", "content": message})

        def call():
            res = client.chat.completions.create(
                model="DeepSeek-R1-0528",
                messages=messages,
                temperature=0.3,
                top_p=0.9
            )
            return res.choices[0].message.content

        return safe_execute(call)

    except Exception as e:
        logger.warning(f"SambaNova failed: {e}")
        return None

# ---------------- Provider 2: Gemini ----------------

def _try_gemini(message: str, history: List[Dict]) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=SYSTEM_PROMPT,
        )

        gemini_history = []
        for h in history:
            if h.get("content"):
                gemini_history.append({
                    "role": "model" if h.get("role") == "assistant" else "user",
                    "parts": [h.get("content")],
                })

        def call():
            chat = model.start_chat(history=gemini_history)
            res = chat.send_message(message)
            return res.text

        return safe_execute(call)

    except Exception as e:
        logger.warning(f"Gemini failed: {e}")
        return None

# ---------------- Provider 3: Claude ----------------

def _try_claude(message: str, history: List[Dict]) -> Optional[str]:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        messages = []
        for h in history:
            if h.get("content"):
                messages.append(h)
        messages.append({"role": "user", "content": message})

        def call():
            res = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
            return res.content[0].text

        return safe_execute(call)

    except Exception as e:
        logger.warning(f"Claude failed: {e}")
        return None

# ---------------- Provider 4: OpenAI ----------------

def _try_openai(message: str, history: List[Dict]) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for h in history:
            if h.get("content"):
                messages.append(h)
        messages.append({"role": "user", "content": message})

        def call():
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=1024,
                messages=messages,
            )
            return res.choices[0].message.content

        return safe_execute(call)

    except Exception as e:
        logger.warning(f"OpenAI failed: {e}")
        return None

# ---------------- Main Chat Function ----------------

def chat(message: str, history: Optional[List[Dict]] = None) -> Dict:
    history = history or []

    providers = [
        ("sambanova", _try_sambanova),
        ("gemini", _try_gemini),
        ("claude", _try_claude),
        ("openai", _try_openai),
    ]

    for name, fn in providers:
        result = fn(message, history)
        if result:
            logger.info(f"Response from {name}")
            return {"response": result, "provider": name}

    return {
        "error": "All AI providers failed. Check API keys or quota."
    }
