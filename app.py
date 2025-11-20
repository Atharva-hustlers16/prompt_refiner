# app.py
"""
Streamlit UI for the robust Prompt Refinement pipeline.
Model selection, persistent history, and Test-optimized-prompt button.
"""

import streamlit as st
import time
import json

from main import (
    run_prompt_refinement_crew,
    test_prompt_with_gemini,
    load_history,
    save_history_item,
    make_history_item
)

st.set_page_config(page_title="Prompt Refinement Crew — Gemini", layout="wide", page_icon="🛠️")
st.title("📝 Prompt Refinement & Optimization Crew — Gemini (robust)")

col_main, col_sidebar = st.columns([3, 1])

with col_main:
    st.markdown("Enter a raw prompt and choose a Gemini model. Click **Refine Prompt** to run the pipeline.")
    raw_prompt = st.text_area("🗒️ Raw prompt", height=160, value="Write a 200-word friendly essay on engineering, divided into short paragraphs.")
    model_choice = st.selectbox("Select Gemini model", options=[
        "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-1.5"
    ], index=0)

    agent_max_tokens = st.slider("Agent max tokens (each agent call)", min_value=128, max_value=800, value=300, step=64)
    test_max_tokens = st.slider("Test response tokens", min_value=64, max_value=1024, value=200, step=32)

    if st.button("🔧 Refine Prompt"):
        if not raw_prompt.strip():
            st.warning("Please enter a raw prompt to refine.")
        else:
            with st.spinner("Running Clarify → Constraints → Synthesize (may take a few seconds)..."):
                start = time.time()
                try:
                    outputs = run_prompt_refinement_crew(raw_prompt, model=model_choice, agent_max_tokens=agent_max_tokens)
                except Exception as e:
                    st.error(f"Error running pipeline: {e}")
                    outputs = None
                elapsed = time.time() - start

            if outputs:
                st.markdown("### ✅ Clarified Intent & Ambiguities")
                st.write(outputs["clarified"])
                st.markdown("### ⚙️ Constraints")
                st.write(outputs["constraints"])
                st.markdown("### ✨ Final Optimized Prompt")
                st.code(outputs["final_prompt"], language="")
                item = make_history_item(raw_prompt, model_choice, agent_max_tokens, outputs, elapsed)
                try:
                    save_history_item(item)
                except Exception as e:
                    st.error(f"Failed to save history: {e}")
                st.success("Prompt refined and saved to history (sidebar).")

with col_sidebar:
    st.header("History")
    history = load_history()
    if not history:
        st.info("No runs yet — refine a prompt to create entries.")
    else:
        selected_id = st.radio(
            "Select run",
            options=[item["id"] for item in history],
            format_func=lambda i: next(x for x in history if x["id"] == i)["created_at"]
        )
        selected_item = next((x for x in history if x["id"] == selected_id), None)

        if selected_item:
            st.markdown(f"**Model:** {selected_item['model']}")
            st.markdown("**Raw prompt**")
            st.write(selected_item["raw_prompt"])
            st.markdown("**Final optimized prompt**")
            st.code(selected_item["final_prompt"], language="")
            st.markdown("**Clarified**")
            st.write(selected_item["clarified"])
            st.markdown("**Constraints**")
            st.write(selected_item["constraints"])
            st.write("---")

            if st.button("▶️ Test optimized prompt (generate sample reply)"):
                with st.spinner("Generating sample reply from Gemini..."):
                    try:
                        sample = test_prompt_with_gemini(selected_item["final_prompt"], model=selected_item["model"], max_output_tokens=test_max_tokens)
                    except Exception as e:
                        st.error(f"Error while testing prompt: {e}")
                        sample = None

                if sample:
                    if isinstance(sample, str) and sample.startswith("DEBUG"):
                        st.markdown("### ⚠️ Debug output (no direct text extracted)")
                        st.code(sample, language="")
                    else:
                        st.markdown("### Sample reply")
                        st.write(sample)

                    # save sample into history
                    history = load_history()
                    for it in history:
                        if it["id"] == selected_item["id"]:
                            it["sample_reply"] = sample
                            it["sample_reply_at"] = time.ctime()
                            break
                    try:
                        with open("history.json", "w", encoding="utf-8") as f:
                            json.dump(history, f, indent=2, ensure_ascii=False)
                        st.success("Saved sample to history.")
                    except Exception as e:
                        st.error(f"Failed to save sample to history: {e}")

    st.markdown("---")
    if st.button("🧾 Download history (JSON)"):
        history = load_history()
        if history:
            st.download_button("Download JSON", data=json.dumps(history, indent=2), file_name="prompt_refinement_history.json", mime="application/json")
        else:
            st.info("History is empty.")

st.caption("Notes: • The app retries automatically once with conservative settings when Gemini returns no text. • Ask for smaller outputs (words, not 'lines') to avoid token limits. • Debug blobs start with DEBUG_ and can be pasted here if needed.")
