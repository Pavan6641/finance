#python ppwritefile personal_finance_chatbot.py
"""
Personal Finance Chatbot - Streamlit single-file app for Google Colab.

Uses:
- Hugging Face Inference API to call IBM Granite (recommended for Colab)
- Optional IBM Watson Assistant (lightweight REST example) - disabled by default

How to run:
- Set HUGGINGFACE_API_TOKEN in environment (.env)
- Run streamlit with ngrok in Colab (see instructions in notebook)
"""

import os
import json
import streamlit as st
import requests
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

# Configuration
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
GRANITE_MODEL = os.getenv("GRANITE_MODEL", "ibm-granite/granite-3.3-2b-instruct")
WATSON_APIKEY = os.getenv("WATSON_APIKEY")
WATSON_URL = os.getenv("WATSON_URL")
WATSON_ASSISTANT_ID = os.getenv("WATSON_ASSISTANT_ID")

# --------- Helper: Hugging Face Inference API call (Granite) ----------
def call_hf_inference(prompt: str, model: str = None, max_tokens: int = 300, temperature: float = 0.2) -> str:
    model = model or GRANITE_MODEL
    if not HUGGINGFACE_API_TOKEN:
        return "ERROR: HUGGINGFACE API token not set."
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "temperature": temperature},
        "options": {"use_cache": False}
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # HF can return list/dict; try to extract generated text robustly
        if isinstance(data, list) and len(data) > 0 and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        # Fallback: stringify
        return json.dumps(data)
    except Exception as e:
        return f"Exception calling Hugging Face Inference API: {str(e)}"

# --------- Optional: Watson Assistant (simple REST example) ----------
def call_watson_assistant(message: str) -> str:
    if not (WATSON_APIKEY and WATSON_URL and WATSON_ASSISTANT_ID):
        return "Watson not configured."
    auth = ("apikey", WATSON_APIKEY)
    headers = {"Content-Type": "application/json"}
    try:
        # Create session
        sess = requests.post(f"{WATSON_URL}/v2/assistants/{WATSON_ASSISTANT_ID}/sessions?version=2024-10-01",
                             auth=auth, headers=headers, timeout=20)
        sess.raise_for_status()
        session_id = sess.json().get("session_id")
        # Send message
        payload = {"input": {"message_type": "text", "text": message}}
        resp = requests.post(f"{WATSON_URL}/v2/assistants/{WATSON_ASSISTANT_ID}/sessions/{session_id}/message?version=2024-10-01",
                             auth=auth, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        out = resp.json()
        # Tear down
        try:
            requests.delete(f"{WATSON_URL}/v2/assistants/{WATSON_ASSISTANT_ID}/sessions/{session_id}?version=2024-10-01",
                            auth=auth, timeout=5)
        except:
            pass
        texts = []
        for r in out.get("output", {}).get("generic", []):
            if r.get("response_type") == "text":
                texts.append(r.get("text", ""))
        return "\\n".join(texts) if texts else json.dumps(out)
    except Exception as e:
        return f"Exception calling Watson Assistant: {str(e)}"

# --------- Prompt engineering helpers ----------
def build_prompt(user_question: str, persona: str = "student") -> str:
    if persona == "student":
        tone = "Clear, friendly, simple. Define financial terms in plain English and give short examples."
        depth = "Provide practical steps a student can follow and a short sample monthly budget table."
    else:
        tone = "Concise, professional, data-forward. Use precise terminology and include numeric examples where helpful."
        depth = "Provide trade-offs and a recommended allocation by percentage if relevant."
    prompt = f"You are a helpful personal finance assistant. {tone} {depth}\\n\\nUser question: {user_question}\\n\\nOutput:"
    return prompt

def generate_budget_summary(income: float, essentials_pct: float = 0.5, savings_pct: float = 0.2) -> str:
    essentials = income * essentials_pct
    savings = income * savings_pct
    discretionary = income - essentials - savings
    return (
        f"Monthly Budget Summary:\\n"
        f"Total income: ₹{income:,.2f}\\n"
        f"Essentials ({essentials_pct*100:.0f}%): ₹{essentials:,.2f}\\n"
        f"Savings ({savings_pct*100:.0f}%): ₹{savings:,.2f}\\n"
        f"Discretionary: ₹{discretionary:,.2f}\\n"
    )

# --------- Streamlit UI ----------
st.set_page_config(page_title="Personal Finance Chatbot", layout="centered")
st.title("Personal Finance Chatbot — Demo (Granite via Hugging Face)")
st.caption("Uses Hugging Face Inference API to call Granite model; optional Watson Assistant.")

with st.sidebar:
    st.header("Settings")
    backend = st.selectbox("Primary backend", ["Granite (HuggingFace Inference API)", "IBM Watson Assistant"])
    persona = st.radio("Persona / Tone", ["student", "professional"])
    st.markdown("---")
    st.write("Environment: (Loaded from .env in Colab)")
    st.write(f"Hugging Face token set: {'✅' if HUGGINGFACE_API_TOKEN else '❌'}")
    st.write(f"Watson configured: {'✅' if (WATSON_APIKEY and WATSON_URL and WATSON_ASSISTANT_ID) else '❌'}")

st.header("Ask your finance question")
user_q = st.text_area("Question", height=140, placeholder="E.g., How should I budget ₹30,000/month as a student?")

col1, col2 = st.columns(2)
with col1:
    income = st.number_input("Optional: Monthly income (₹)", min_value=0.0, value=0.0, step=1000.0)
with col2:
    optimize_for = st.selectbox("Optimize for", ["savings", "growth", "safety"])

if st.button("Get Answer"):
    if not user_q.strip():
        st.warning("Please enter a question.")
    else:
        st.info("Generating response — this may take a few seconds.")
        prompt = build_prompt(user_q, persona)
        if income > 0:
            prompt += "\\n\\nReference budget (income info):\\n" + generate_budget_summary(income)

        if backend.startswith("Granite"):
            hf_resp = call_hf_inference(prompt, model=GRANITE_MODEL, max_tokens=400, temperature=0.2)
            st.subheader("Granite (Hugging Face) response")
            st.write(hf_resp)
        else:
            watson_resp = call_watson_assistant(prompt)
            st.subheader("Watson Assistant response")
            st.write(watson_resp)

        if income > 0:
            st.markdown("---")
            st.subheader("Local budget summary")
            st.code(generate_budget_summary(income))

st.markdown("---")
st.write("Developer notes: Replace placeholders and secure your tokens. For production, add authentication and secure storage.")
