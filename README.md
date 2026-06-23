---
title: Symptom Triage Chatbot
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: mit
---

# Symptom Triage Chatbot

A free symptom triage chatbot using:

- Python 3.12.4
- Streamlit
- LangChain
- Mistral AI
- DuckDuckGo search through ddgs
- Hugging Face Spaces Docker hosting

## Important safety note

This app does not provide a confirmed diagnosis. It only gives educational symptom triage guidance.

For severe symptoms, seek urgent medical care.

## Required secret

Add this secret in Hugging Face Spaces or LLM Api keys:

```txt
MISTRAL_API_KEY