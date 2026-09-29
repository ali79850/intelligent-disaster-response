"""
Phase 11: LLM narrative summary generator (Groq free tier).

Strict constraint, enforced by design: the LLM receives ONLY the already-
computed structured report (from src/inference/report.py) and narrates it
in prose. It must never invent facts not present in the input - no
casualties, no population figures, no infrastructure status, no rescue
requirements. All factual numbers originate from the actual ML pipeline
output, never from the LLM itself.
"""
from datetime import datetime, timezone
import json
import os

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are a report-writing assistant for a disaster damage \
assessment tool. You will be given a JSON object containing the ONLY facts \
you are allowed to state. Your job is to write a short, clear, professional \
paragraph (4-6 sentences) summarizing these exact facts in plain language \
for a non-technical reader.

STRICT RULES, NO EXCEPTIONS:
- Use ONLY the numbers and facts present in the provided JSON. Do not round \
differently, do not estimate, do not add context not present in the data.
- NEVER mention casualties, deaths, injuries, population, number of people \
affected, rescue requirements, or infrastructure operational status (e.g. \
hospitals, roads) - none of this data exists in the input, and stating it \
would be fabrication.
- NEVER claim certainty the data doesn't support. If requires_human_review \
is true, say so plainly and explain why (which classes are low-confidence).
- Always include the disclaimer that this is a preliminary, AI-generated \
assessment, not an authoritative determination - use the limitations field \
provided, do not write your own version of it.
- Do not speculate about the cause of damage, the disaster type, or any \
recommended actions beyond what "requires human review" implies.
- If you are unsure whether something is supported by the data, leave it \
out rather than guess.
"""

DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def _generate_groq_summary(report: dict, model: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    from groq import Groq

    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to .env (e.g. GROQ_API_KEY=gsk_...) or provide it in the request."
        )

    model_name = model or DEFAULT_GROQ_MODEL
    client = Groq(api_key=key)

    user_message = (
        "Here is the structured damage assessment report. Write the summary "
        "paragraph now, following the rules exactly.\n\n"
        f"{json.dumps(report, indent=2, default=str)}"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1024,
        temperature=0.2,
        reasoning_effort="low",
        extra_body={"reasoning_format": "hidden"},
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(
            f"Groq returned empty content for model {model_name}. "
            f"Full response: {response.model_dump()}"
        )
    return content.strip(), model_name


def _generate_gemini_summary(report: dict, model: str | None = None, api_key: str | None = None) -> tuple[str, str]:
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise RuntimeError(
            "google-genai package is not installed. Install it with: pip install google-genai"
        )

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to .env (e.g. GEMINI_API_KEY=AIza...) or provide it in the request."
        )

    model_name = model or DEFAULT_GEMINI_MODEL
    client = genai.Client(api_key=key)

    user_message = (
        "Here is the structured damage assessment report. Write the summary "
        "paragraph now, following the rules exactly.\n\n"
        f"{json.dumps(report, indent=2, default=str)}"
    )

    response = client.models.generate_content(
        model=model_name,
        contents=user_message,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )

    content = response.text or ""
    return content.strip(), model_name


def generate_narrative_summary(
    report: dict,
    provider: str = "groq",
    model: str | None = None,
    api_key: str | None = None,
    return_metadata: bool = False,
) -> str | dict:
    """
    Generates a disciplined, non-fabricating narrative summary of a structured
    damage assessment report using an LLM API (Groq or Google Gemini).
    """
    prov = (provider or "groq").lower().strip()

    if prov == "groq":
        content, used_model = _generate_groq_summary(report, model=model, api_key=api_key)
    elif prov in ("gemini", "google"):
        content, used_model = _generate_gemini_summary(report, model=model, api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider '{provider}'. Must be 'groq' or 'gemini'.")

    if return_metadata:
        return {
            "summary": content,
            "provider": prov,
            "model": used_model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
    return content