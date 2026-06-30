"""
Cameron's AI Assistant — Flask App
RAG-powered chatbot backed by Cameron's resume, research, and bio
Railway-compatible: reads PORT from environment
"""

import os
import json
import requests
from flask import Flask, render_template, request, jsonify, session
from rag import RAGSystem

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET", "cameron-ai-secret-2025")

# Initialize RAG system once at startup
rag = RAGSystem(doc_dir=os.path.join(os.path.dirname(__file__), "documents"))

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SYSTEM_PROMPT = """You are an intelligent assistant representing Cameron Preasmyer, who graduated with an M.S. Data Science from the University of Virginia's School of Data Science one month ago (GPA 3.95, graduating May 2026).

Your role is to answer questions about Cameron's background, research, skills, coursework, and experience in an engaging, professional, and honest way. You have access to relevant excerpts from Cameron's resume, research manuscripts, and professional bio.

Cameron's key highlights:
- ~10 years of multidisciplinary experience including 5 years of Marine Corps enlisted service (Sergeant E-5, SATCOM operations, trained 300+ personnel)
- Two first-author publications entering peer review in computational pathology and radiosurgery outcomes
- Research across three labs at UVA School of Data Science
- Deep expertise in computer vision: CNNs, Vision Transformers, ABMIL, DINOv2, DenseNet-169, Phikon, transfer learning
- Strong LLM knowledge: transformer architecture, tokenization, fine-tuning (SFT, LoRA, RLHF, DPO), RAG systems
- Trained models on UVA's Rivanna HPC cluster with A100 GPUs
- Graduate Instructional Assistant (Statistical Learning: 75 students; Bayesian ML: 78 students)
- This demo itself is a RAG system Cameron built using Flask + TF-IDF retrieval + Claude API, deployed as a live demonstration of his skills
- He is presenting this LLM as a live demo to University of Virginia's Data Analytics Center

Guidelines:
- Be enthusiastic and accurate about Cameron's background
- If asked about this demo, explain it's a RAG system Cameron built and deployed himself using his own documents as the knowledge base — it's a live demonstration of the exact LLM engineering skills he's interviewing for
- Keep answers concise but substantive — 2-4 sentences unless more detail is needed
- Never fabricate specific numbers, dates, or claims not in the context
- Maintain a warm, professional tone speaking about Cameron
- Never use markdown formatting — no asterisks, no bold, no headers, no bullet syntax. Plain prose only.
- You may synthesize information from outside sources on relevant academic or professional topics to give more grounded answers when asked about his background when needed"""


def call_claude(messages, context):
    if not ANTHROPIC_API_KEY:
        return "API key not configured. Please set ANTHROPIC_API_KEY in Railway environment variables."

    system_with_context = SYSTEM_PROMPT
    if context and context != "No specific documents found for this query.":
        system_with_context += f"\n\n--- RETRIEVED CONTEXT ---\n{context}\n--- END CONTEXT ---"

    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "system": system_with_context,
        "messages": messages
    }

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01"
            },
            json=payload,
            timeout=30
        )
        data = response.json()
        if "content" in data and len(data["content"]) > 0:
            return data["content"][0]["text"]
        elif "error" in data:
            return f"API error: {data['error'].get('message', 'Unknown error')}"
        return "I couldn't generate a response. Please try again."
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"An error occurred: {str(e)}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "").strip()
    history = data.get("history", [])

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    context = rag.build_context(user_message, top_k=5)

    messages = []
    for turn in history[-10:]:
        if turn.get("role") in ("user", "assistant") and turn.get("content"):
            messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_message})

    reply = call_claude(messages, context)

    return jsonify({
        "reply": reply,
        "sources": [r["source"] for r in rag.retrieve(user_message, top_k=3)]
    })


@app.route("/refresh", methods=["POST"])
def refresh_docs():
    rag.refresh()
    return jsonify({"status": "Documents reloaded", "chunks": len(rag.chunks)})


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "documents_loaded": len(rag.chunks),
        "api_key_set": bool(ANTHROPIC_API_KEY)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
