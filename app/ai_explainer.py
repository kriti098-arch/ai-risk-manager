"""
AI-generated natural-language explanations, using Google's Gemini API
(free tier -- see aistudio.google.com). Uses the current `google-genai`
SDK (the older `google-generativeai` package is deprecated as of 2026).

Why this exists: the SHAP bar chart is accurate but requires the viewer
to already understand what "SHAP impact" means. This module takes the
SAME structured SHAP output already computed by the model and asks an
LLM to turn it into one clear paragraph a non-technical reviewer can
read directly -- e.g. a fraud analyst, a merchant, or a judge who isn't
an ML engineer.

This is a genuinely separate AI system layered ON TOP of the fraud
model, not a replacement for it: the Random Forest + Isolation Forest
ensemble still makes the actual risk decision. This module only
explains that decision in plain language -- it never independently
decides fraud/not-fraud, so it can't override or contradict the
underlying model's decision, only describe it.

Timeout handling: the google-genai SDK has known issues where its own
timeout configuration doesn't reliably prevent indefinite hangs (the
underlying httpx client can ignore it). Rather than trust the SDK's
timeout alone, this module enforces its OWN hard deadline using a
background thread + explicit result(timeout=...), and abandons the
thread (shutdown(wait=False)) if it's still running past the deadline
-- so a slow or hung network call degrades to a clear error message
in the dashboard instead of freezing the whole app.
"""
import concurrent.futures

from google import genai

MODEL_NAME = "gemini-2.5-flash"  # free tier, fast, cheap -- right fit for a short explanation task


def _build_prompt(transaction_values: dict, risk_score: float, decision: str,
                   shap_reasons: list, pattern_hint: str) -> str:
    reasons_text = "\n".join(
        f"- {r['feature']}: SHAP impact {r['impact']:+.3f} "
        f"({'pushes risk UP' if r['impact'] > 0 else 'pushes risk DOWN'}), "
        f"actual value = {transaction_values.get(r['feature'], 'n/a')}"
        for r in shap_reasons
    )
    return f"""You are explaining a fraud-detection model's decision to a non-technical reader (a fraud analyst or a business reviewer, not a data scientist).

Decision: {decision}
Risk score: {risk_score:.1%}
System's own pattern guess: {pattern_hint}

Top contributing factors (from SHAP, a model-explainability technique):
{reasons_text}

Write ONE short paragraph (3-4 sentences, plain English, no jargon like "SHAP" or "z-score") explaining why this transaction likely got this decision. Be concrete -- reference the actual factors above. Do not invent any facts not given above. Do not add a recommendation or disclaimer -- just the explanation."""


def generate_ai_explanation(transaction_values: dict, risk_score: float, decision: str,
                             shap_reasons: list, pattern_hint: str, api_key: str,
                             timeout_seconds: int = 12) -> str:
    """Returns a plain-English explanation string, or a clear error message
    (never raises) so a missing/invalid key, a network hiccup, or a slow/
    hung response always degrades gracefully instead of freezing the app."""
    if not api_key:
        return ("⚠️ No Gemini API key configured. Add GEMINI_API_KEY to your "
                "Streamlit secrets to enable AI-generated explanations.")

    prompt = _build_prompt(transaction_values, risk_score, decision, shap_reasons, pattern_hint)

    def _call():
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return (response.text or "").strip()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_call)
    try:
        text = future.result(timeout=timeout_seconds)
        executor.shutdown(wait=False)
        return text if text else "⚠️ Gemini returned an empty response -- try again."
    except concurrent.futures.TimeoutError:
        executor.shutdown(wait=False)  # don't block the app waiting for a hung call
        return f"⚠️ Gemini didn't respond within {timeout_seconds}s. The chart above still shows the real SHAP explanation regardless."
    except Exception as e:
        executor.shutdown(wait=False)
        return f"⚠️ Couldn't reach Gemini right now ({type(e).__name__}). The chart above still shows the real SHAP explanation regardless."