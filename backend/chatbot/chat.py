"""
Agriculture chatbot using OpenAI or Gemini.
API key from .env; returns error if missing.
"""
import os

# Prefer OpenAI; fallback to Gemini if OPENAI_API_KEY not set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

SYSTEM_PROMPT = """You are KrishiSahyog AI, a helpful agriculture assistant for Indian farmers.
You help with:
- Plant disease identification and treatment
- Soil analysis and fertilizer recommendations
- Crop selection and planting advice
- Weather and irrigation guidance
- Pest and weed management
- Market and pricing insights

Respond in clear, simple language. Use Hindi or English based on the user's preference.
Be practical and specific. When giving advice, consider Indian farming conditions."""


def chat(user_message: str, history: list[dict] | None = None) -> dict:
    """
    Send message to LLM and get response.
    Returns: {response, error?, provider}
    """
    history = history or []

    # Try OpenAI first
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            for h in history[-10:]:  # Last 10 exchanges
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
            messages.append({"role": "user", "content": user_message})

            completion = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
            )
            content = completion.choices[0].message.content or ""
            return {"response": content, "provider": "openai"}
        except Exception as e:
            return {"response": "", "error": str(e), "provider": "openai"}

    # Try Gemini
    if GEMINI_API_KEY:
        try:
            try:
                import google.generativeai as genai
            except ImportError:
                return {
                    "response": "",
                    "error": "Gemini SDK not installed. Run: pip install google-generativeai",
                    "provider": None,
                }
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-pro")
            chat_session = model.start_chat(history=[])
            for h in history[-10:]:
                if h.get("role") == "user":
                    chat_session.send_message(h.get("content", ""))
                else:
                    pass  # Gemini handles history differently
            result = chat_session.send_message(user_message)
            return {"response": result.text or "", "provider": "gemini"}
        except Exception as e:
            return {"response": "", "error": str(e), "provider": "gemini"}

    return {
        "response": "",
        "error": "No LLM API key configured. Set OPENAI_API_KEY or GEMINI_API_KEY in backend/.env",
        "provider": None,
    }
