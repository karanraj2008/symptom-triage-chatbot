import os
import certifi
from typing import List, Dict, Any

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field


from ddgs import DDGS

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "mistral-small-latest")


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Symptom Triage Chatbot",
    page_icon="🩺",
    layout="centered"
)


# -----------------------------
# Output schema
# -----------------------------
class PossibleCause(BaseModel):
    condition: str = Field(description="Possible condition, not confirmed diagnosis")
    why_it_may_fit: str = Field(description="Short explanation based on symptoms")
    urgency: str = Field(description="low, medium, high, or emergency")
    confidence: str = Field(description="low, medium, or high")


class TriageOutput(BaseModel):
    assistant_message: str = Field(description="Friendly explanation for user")
    emergency: bool = Field(description="True if emergency care may be needed")
    possible_causes: List[PossibleCause] = Field(description="Educational possibilities only")
    follow_up_questions: List[str] = Field(description="Questions to clarify symptoms")
    recommended_specialty: str = Field(description="Doctor specialty to consult")
    warning_signs: List[str] = Field(description="Urgent warning signs")
    safe_general_steps: List[str] = Field(description="Safe general non-prescription steps")
    doctor_needed: bool = Field(description="Whether doctor consultation is suggested")


parser = JsonOutputParser(pydantic_object=TriageOutput)


# -----------------------------
# Medical safety prompt
# -----------------------------
SYSTEM_TEMPLATE = """
You are a careful medical symptom triage assistant.

Your job:
- Understand symptoms described by the user.
- Provide educational possibilities, not a confirmed diagnosis.
- Ask follow-up questions if required.
- Recommend the right doctor specialty.
- Mention warning signs.
- Never prescribe medicine or dosage.
- Never say the user definitely has a disease.

Important safety rules:
1. You are not a doctor.
2. Always mention this is not a medical diagnosis.
3. If symptoms suggest emergency risk, set emergency=true.
4. If emergency=true, tell the user to seek urgent/emergency medical help immediately.
5. Do not provide medicine names, dosage, or treatment plans.
6. Do not discuss sexual details. If symptoms are private/sensitive, advise consulting a qualified doctor directly.

Emergency examples:
- chest pain or pressure
- severe breathing difficulty
- fainting
- confusion
- sudden weakness or numbness on one side
- seizure
- severe bleeding
- severe allergic reaction
- very high fever with stiff neck
- severe abdominal pain
- suicidal or self-harm thoughts

Return ONLY valid JSON using this format:
{format_instructions}

Conversation history:
{history}

User message:
{user_message}
"""


# -----------------------------
# LangChain + Mistral
# -----------------------------
def get_llm_chain():
    if not MISTRAL_API_KEY:
        return None

    llm = ChatMistralAI(
        api_key=MISTRAL_API_KEY,
        model=MODEL_NAME,
        temperature=0.2,
        max_retries=2,
    )

    prompt = PromptTemplate(
        template=SYSTEM_TEMPLATE,
        input_variables=["history", "user_message"],
        partial_variables={
            "format_instructions": parser.get_format_instructions()
        },
    )

    return prompt | llm | parser


# -----------------------------
# Chat history helper
# -----------------------------
def build_history_text(messages: List[Dict[str, str]], limit: int = 8) -> str:
    recent = messages[-limit:]
    history_lines = []

    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        history_lines.append(f"{role.upper()}: {content}")

    return "\n".join(history_lines)


# -----------------------------
# Free doctor search using DuckDuckGo
# -----------------------------
def search_doctors_free(
    specialty: str,
    location: str,
    max_results: int = 5
) -> List[Dict[str, str]]:
    """
    Free live doctor search using DuckDuckGo through ddgs.
    No API key required.
    """

    query = (
        f"best {specialty} doctor near {location} "
        f"experience fee consultation clinic"
    )

    results = []

    try:
        with DDGS() as ddgs:
            search_results = ddgs.text(
                query,
                region="in-en",
                safesearch="moderate",
                max_results=max_results
            )

            for item in search_results:
                results.append({
                    "title": item.get("title", ""),
                    "href": item.get("href", ""),
                    "body": item.get("body", "")
                })

    except Exception as e:
        st.warning(f"Search failed: {e}")

    return results


# -----------------------------
# UI render helpers
# -----------------------------
def render_triage_result(result: Dict[str, Any]):
    emergency = result.get("emergency", False)

    if emergency:
        st.error("🚨 This may need urgent/emergency care. Please seek immediate medical help.")
    else:
        st.info("ℹ️ This is educational guidance only, not a medical diagnosis.")

    st.write(result.get("assistant_message", ""))

    possible_causes = result.get("possible_causes", [])
    if possible_causes:
        with st.expander("Possible causes"):
            for cause in possible_causes:
                st.markdown(f"**{cause.get('condition', 'Unknown')}**")
                st.markdown(f"- Why it may fit: {cause.get('why_it_may_fit', '')}")
                st.markdown(f"- Urgency: `{cause.get('urgency', '')}`")
                st.markdown(f"- Confidence: `{cause.get('confidence', '')}`")

    follow_up_questions = result.get("follow_up_questions", [])
    if follow_up_questions:
        with st.expander("Follow-up questions"):
            for question in follow_up_questions:
                st.markdown(f"- {question}")

    warning_signs = result.get("warning_signs", [])
    if warning_signs:
        with st.expander("Warning signs"):
            for sign in warning_signs:
                st.markdown(f"- {sign}")

    safe_steps = result.get("safe_general_steps", [])
    if safe_steps:
        with st.expander("General safe steps"):
            for step in safe_steps:
                st.markdown(f"- {step}")

    specialty = result.get("recommended_specialty", "General Physician")
    st.success(f"Suggested specialist: **{specialty}**")


def render_search_results(results: List[Dict[str, str]]):
    st.subheader("Doctor search results")

    if not results:
        st.warning("No search results found. Try a more specific area or city.")
        return

    st.caption(
        "These are live web search results. Please verify doctor qualification, "
        "fee, experience, reviews, and availability before booking."
    )

    for item in results:
        title = item.get("title", "Result")
        href = item.get("href", "")
        body = item.get("body", "")

        with st.container(border=True):
            st.markdown(f"### {title}")
            st.write(body)

            if href:
                st.markdown(f"[Open result]({href})")


# -----------------------------
# Main UI
# -----------------------------
st.title("🩺 Symptom Triage Chatbot")
st.caption(
    "Describe your symptoms. The app suggests possible causes, asks follow-up questions, "
    "and searches doctors online for free."
)

with st.sidebar:
    st.header("Location")

    location = st.text_input(
        "Enter your area/city",
        value="Indiranagar Bengaluru",
        help="Example: Indiranagar Bengaluru, Koramangala Bengaluru, Whitefield Bengaluru"
    )

    st.markdown("---")

    st.warning(
        "This app does not provide a confirmed diagnosis. "
        "For serious or worsening symptoms, seek urgent medical care."
    )

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.last_triage = None
        st.session_state.search_results = []
        st.rerun()


# -----------------------------
# Check API key
# -----------------------------
if not MISTRAL_API_KEY:
    st.error(
        "MISTRAL_API_KEY is missing. Add it in `.env` locally or Hugging Face Space secrets."
    )
    st.stop()


# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_triage" not in st.session_state:
    st.session_state.last_triage = None

if "search_results" not in st.session_state:
    st.session_state.search_results = []


# -----------------------------
# Initial chatbot message
# -----------------------------
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Hi! Tell me your symptoms, how long you have had them, severity, "
            "your age range, and anything that makes them better or worse."
        )
    })


# -----------------------------
# Render chat messages
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Describe your symptoms...")

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)

    chain = get_llm_chain()

    if chain is None:
        st.error("Could not initialize Mistral AI. Check your API key.")
        st.stop()

    history = build_history_text(st.session_state.messages)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chain.invoke({
                    "history": history,
                    "user_message": user_input
                })

                st.session_state.last_triage = result
                st.session_state.search_results = []

                assistant_text = result.get(
                    "assistant_message",
                    "I could not process that clearly. Please describe your symptoms again."
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_text
                })

                render_triage_result(result)

            except Exception as e:
                error_message = (
                    "Sorry, I could not process that safely. "
                    "Please try again with symptoms, duration, severity, and age range."
                )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

                st.error(error_message)
                st.caption(str(e))


# -----------------------------
# Doctor search section
# -----------------------------
if st.session_state.last_triage:
    triage = st.session_state.last_triage

    st.markdown("---")

    if triage.get("emergency", False):
        st.error(
            "Because this may be urgent, please do not wait for a normal appointment. "
            "Seek emergency medical care immediately."
        )
    else:
        specialty = triage.get("recommended_specialty", "General Physician")

        st.subheader("Find doctors online")

        if st.button(f"Search {specialty} doctors near {location}"):
            with st.spinner("Searching doctors online..."):
                st.session_state.search_results = search_doctors_free(
                    specialty=specialty,
                    location=location,
                    max_results=5
                )

        if st.session_state.search_results:
            render_search_results(st.session_state.search_results)