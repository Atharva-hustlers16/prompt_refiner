# main.py
"""
Prompt Refinement pipeline (robust Gemini usage with automatic retry).
- Uses google.generativeai (Gemini)
- Explicit contents=[prompt] calls
- Automatic retry with conservative settings if the model returns no text
- Returns debug blobs prefixed with DEBUG_ when extraction fails
- History persistence (history.json)
"""

import os
import json
import time
import traceback
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from time import sleep

# Load .env
load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not GEMINI_KEY:
    raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) not found in .env")

# Configure SDK
genai.configure(api_key=GEMINI_KEY)

HISTORY_FILE = "history.json"

# ---------- prompt templates ----------
CLARITY_PROMPT = """You are the Clarity Analyst.
User raw prompt:
\"\"\"{raw}\"\"\"

Task: Identify ambiguous parts, unspecified goals, hidden assumptions, and the user's true intent.
Return:
1) Intent summary (one short sentence).
2) Ambiguities / missing info (bullet list).
3) Suggested assumptions (key=value pairs) if required.
"""

CONSTRAINT_PROMPT = """You are the Constraint Engineer.
Input: Clarified intent and suggested assumptions:
\"\"\"{clarified}\"\"\"

Task: Produce explicit constraints and formatting instructions to make a prompt deterministic and high-yield.
Include: output format, tone, length, required sections/headings, persona (single-line system instruction), and one short example output.
Return: a concise constraints block.
"""

SYNTHESIS_PROMPT = """You are the Optimization Synthesizer.
Inputs:
- Raw prompt: \"\"\"{raw}\"\"\"
- Clarifications: \"\"\"{clarified}\"\"\"
- Constraints: \"\"\"{constraints}\"\"\"

Task: Combine the above into a single optimized prompt for an LLM.
Requirements:
 - Start with a single-line system instruction (e.g., "You are a ...").
 - Include explicit Objectives, Required Sections, and an OUTPUT FORMAT block showing exactly how to respond.
 - Keep the final prompt focused and ready-to-use by another LLM.
Return only the final optimized prompt text (no commentary).
"""

# ---------- low-level extraction helpers ----------
def _extract_text_from_resp(resp: Any) -> str:
    """Try multiple possible response shapes and return first meaningful text found."""
    # 1) .text
    try:
        if hasattr(resp, "text") and resp.text:
            return resp.text
    except Exception:
        pass

    # 2) resp.output[0].content[0].text
    try:
        out = getattr(resp, "output", None)
        if out and len(out) > 0:
            first = out[0]
            content = getattr(first, "content", None)
            if content and len(content) > 0:
                c0 = content[0]
                if hasattr(c0, "text") and c0.text:
                    return c0.text
    except Exception:
        pass

    # 3) resp.candidates ...
    try:
        candidates = getattr(resp, "candidates", None)
        if candidates and len(candidates) > 0:
            cand0 = candidates[0]
            if hasattr(cand0, "output_text") and cand0.output_text:
                return cand0.output_text
            if hasattr(cand0, "text") and cand0.text:
                return cand0.text
            if hasattr(cand0, "content") and len(cand0.content) > 0:
                c0 = cand0.content[0]
                if hasattr(c0, "text") and c0.text:
                    return c0.text
    except Exception:
        pass

    # 4) try to convert to dict and search for first string
    try:
        resp_dict = resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)
    except Exception:
        try:
            return str(resp)
        except Exception:
            pass

    def walk(o):
        if isinstance(o, str) and o.strip():
            return o
        if isinstance(o, dict):
            for v in o.values():
                r = walk(v)
                if r:
                    return r
        if isinstance(o, list):
            for v in o:
                r = walk(v)
                if r:
                    return r
        return None

    found = walk(resp_dict)
    if found:
        return found

    raise ValueError("No textual content found in response.")


# ---------- robust caller with one automatic retry ----------
def gemini_generate_with_retry(prompt: str, model: str, max_output_tokens: int = 512, debug: bool = False) -> str:
    """
    Call Gemini and automatically retry once with conservative settings if no nontrivial text is produced.
    Returns extracted text or a DEBUG_ blob when nothing usable can be extracted.
    """
    def call_generate(contents_prompt: str, cfg: Dict) -> (Optional[str], Any):
        """Call generate_content and attempt to extract text. Returns (text or None, raw_resp)."""
        try:
            model_obj = genai.GenerativeModel(model)
            resp = model_obj.generate_content(contents=[contents_prompt], generation_config=cfg)
        except Exception as e:
            return None, {"call_error": str(e), "trace": traceback.format_exc()}

        try:
            text = _extract_text_from_resp(resp)
            return text, resp
        except Exception:
            return None, resp

    # try 1: original parameters
    cfg1 = {"max_output_tokens": max_output_tokens, "temperature": 0.7, "candidate_count": 1}
    text1, resp1 = call_generate(prompt, cfg1)
    if text1 and _is_nontrivial_text(text1, model):
        return text1

    # try 2: conservative retry
    try:
        sleep(0.25)
        cfg2 = {"max_output_tokens": min(max_output_tokens, 256), "temperature": 0.2, "candidate_count": 1}
        text2, resp2 = call_generate(prompt, cfg2)
        if text2 and _is_nontrivial_text(text2, model):
            return text2
    except Exception:
        resp2 = None
        text2 = None

    # nothing usable found — prepare debug blob
    debug_obj = {
        "warning": "no_generated_text_after_retry",
        "model": model,
        "short_prompt": (prompt[:200] + "...") if len(prompt) > 200 else prompt,
        "first_response_preview": _safe_resp_preview(resp1),
        "retry_response_preview": _safe_resp_preview(resp2) if 'resp2' in locals() else None
    }
    if debug:
        return "DEBUG_NO_TEXT_AFTER_RETRY: " + json.dumps(debug_obj, indent=2, default=str)
    raise RuntimeError("Gemini returned no generated text after retry. Debug: " + json.dumps(debug_obj, default=str))


def _is_nontrivial_text(text: str, model_name: str) -> bool:
    """Return True if text looks like real generated output (not echo like 'model')."""
    if not isinstance(text, str):
        return False
    t = text.strip()
    if len(t) < 12:
        return False
    if t.lower() in {model_name.lower(), "model", "gemini"}:
        return False
    return True


def _safe_resp_preview(resp: Any) -> Any:
    """Return a small, serializable preview of resp for debug objects."""
    if resp is None:
        return None
    try:
        if isinstance(resp, dict):
            return {k: (repr(v)[:200] + "...") for k, v in resp.items()}
        if hasattr(resp, "to_dict"):
            d = resp.to_dict()
            # remove huge fields; keep summary
            return {"to_dict_sample": _shallow_trim(d)}
        return repr(resp)[:400]
    except Exception:
        try:
            return repr(resp)[:400]
        except Exception:
            return "<unserializable response>"


def _shallow_trim(d: Any, max_chars: int = 400) -> Any:
    """Trim a dict/list to shallow preview sizes for debug output."""
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            try:
                s = json.dumps(v) if not isinstance(v, (dict, list)) else str(type(v))
                out[k] = s[:max_chars]
            except Exception:
                out[k] = repr(v)[:max_chars]
        return out
    if isinstance(d, list):
        return [str(type(x)) for x in d[:5]]
    return str(d)[:max_chars]


# ---------- pipeline using the robust caller ----------
def run_prompt_refinement_crew(raw_prompt: str, model: str = "gemini-2.5-flash", agent_max_tokens: int = 400) -> Dict[str, str]:
    """
    Run Clarify -> Constraints -> Synthesis using gemini_generate_with_retry.
    Returns {'clarified', 'constraints', 'final_prompt'}
    """
    clarify_in = CLARITY_PROMPT.format(raw=raw_prompt)
    clarified = gemini_generate_with_retry(clarify_in, model=model, max_output_tokens=agent_max_tokens, debug=True)

    constraints_in = CONSTRAINT_PROMPT.format(clarified=clarified)
    constraints = gemini_generate_with_retry(constraints_in, model=model, max_output_tokens=agent_max_tokens, debug=True)

    synth_in = SYNTHESIS_PROMPT.format(raw=raw_prompt, clarified=clarified, constraints=constraints)
    final_prompt = gemini_generate_with_retry(synth_in, model=model, max_output_tokens=agent_max_tokens, debug=True)

    return {"clarified": clarified.strip(), "constraints": constraints.strip(), "final_prompt": final_prompt.strip()}


def test_prompt_with_gemini(final_prompt: str, model: str = "gemini-2.5-flash", max_output_tokens: int = 512) -> str:
    """Run final prompt and return sample; debug=True to get debug blob when parsing fails."""
    return gemini_generate_with_retry(final_prompt, model=model, max_output_tokens=max_output_tokens, debug=True)


# ---------- history helpers ----------
def load_history() -> List[Dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            return []
    return []


def save_history_item(item: Dict) -> None:
    history = load_history()
    history.insert(0, item)
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Warning: failed to save history:", e)


def make_history_item(raw_prompt: str, model: str, agent_max_tokens: int, outputs: Dict[str, str], elapsed_seconds: float) -> Dict:
    return {
        "id": int(time.time() * 1000),
        "created_at": time.ctime(),
        "model": model,
        "agent_max_tokens": agent_max_tokens,
        "raw_prompt": raw_prompt,
        "clarified": outputs.get("clarified"),
        "constraints": outputs.get("constraints"),
        "final_prompt": outputs.get("final_prompt"),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "sample_reply": None
    }


# quick CLI test
if __name__ == "__main__":
    sample = "Write a 200-word essay on engineering in a friendly tone."
    out = run_prompt_refinement_crew(sample, model="gemini-2.5-flash", agent_max_tokens=300)
    print("Clarified:\n", out["clarified"])
    print("Constraints:\n", out["constraints"])
    print("Final Prompt:\n", out["final_prompt"])
