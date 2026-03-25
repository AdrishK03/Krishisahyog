import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """You are KrishiSahyog AI, a helpful agriculture assistant for Indian farmers.
Respond in clear, simple language. Use Hindi or English based on the user's preference.
Be practical and specific. When giving advice, consider Indian farming conditions."""

def chat(user_message: str, history: list[dict] | None = None) -> dict:
    history = history or []

    # =========================
    # ✅ GEMINI (PRIMARY)
    # =========================
    if GEMINI_API_KEY:
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=GEMINI_API_KEY)

            # Gemini expects "model" instead of "assistant"
            contents = []
            for h in history[-10:]:
                role = "model" if h.get("role") == "assistant" else "user"
                contents.append({
                    "role": role,
                    "parts": [{"text": h.get("content", "")}]
                })

            contents.append({"role": "user", "parts": [{"text": user_message}]})

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
            )

            return {
                "response": response.text or "",
                "provider": "gemini"
            }
        except Exception as e:
            print(f"Gemini Error: {e}")
            # If Gemini fails, we don't return yet; we let it fall through to OpenAI
            pass 

    # =========================
    # ✅ OPENAI (FALLBACK)
    # =========================
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in history[-10:]:
                messages.append({
                    "role": h.get("role", "user"),
                    "content": h.get("content", "")
                })
            messages.append({"role": "user", "content": user_message})

            completion = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
            )

            return {
                "response": completion.choices[0].message.content or "",
                "provider": "openai"
            }
        except Exception as e:
            return {"response": "", "error": str(e), "provider": "openai"}

    return {
        "response": "",
        "error": "No API keys configured.",
        "provider": None,
    }