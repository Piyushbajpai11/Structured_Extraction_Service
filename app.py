"""Local Streamlit interface for the structured extraction service."""

import json
import time
from urllib.parse import urlparse

import streamlit as st

from main import (
    MAX_MESSAGE_CHARS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    InputValidationError,
    extract_message,
)


st.set_page_config(
    page_title="Structured Extraction Service",
    page_icon=":material/auto_awesome:",
    layout="centered",
)

st.title("Structured extraction", icon=":material/auto_awesome:")
st.caption(
    "Turn one customer message into a consistent, reviewable record."
)

endpoint_host = urlparse(OLLAMA_BASE_URL).hostname
uses_local_endpoint = endpoint_host in {"localhost", "127.0.0.1", "::1"}

with st.sidebar:
    st.header("About this app")
    st.write("This project extracts key details from one customer message.")
    st.caption(f"Model: `{OLLAMA_MODEL}`")
    if uses_local_endpoint:
        st.success("Inference is configured for Ollama on this computer.")
        st.caption("No hosted model API is used. Electricity cost is not measured.")
    else:
        st.info("Inference uses the configured Ollama server.")
        st.caption("Check that server's data and billing settings.")

with st.container(border=True):
    st.subheader("Enter a message")
    st.write("Paste the message text below. The app checks it before model inference.")

    with st.form("extraction_form"):
        message = st.text_area(
            "Customer message",
            height=200,
            max_chars=MAX_MESSAGE_CHARS,
            placeholder="Example: Hi, I’m looking for pricing for our team...",
            help=f"Maximum length: {MAX_MESSAGE_CHARS:,} characters.",
        )
        submitted = st.form_submit_button(
            "Extract details",
            type="primary",
            icon=":material/arrow_forward:",
        )

if submitted:
    st.session_state.pop("latest_result", None)
    st.session_state.pop("latest_elapsed_ms", None)
    st.session_state.pop("latest_usage", None)
    started_at = time.perf_counter()
    usage_events = []

    try:
        with st.spinner("Reading the message with the local model..."):
            result = extract_message(message, usage_callback=usage_events.append)
    except InputValidationError as exc:
        st.error(f"Input rejected: {exc}", icon=":material/error:")
    except (RuntimeError, ValueError) as exc:
        st.error(f"Extraction failed: {exc}", icon=":material/error:")
    else:
        st.session_state["latest_result"] = result.model_dump(mode="json")
        if usage_events:
            st.session_state["latest_usage"] = usage_events[-1]
        st.session_state["latest_elapsed_ms"] = (
            time.perf_counter() - started_at
        ) * 1000

if "latest_result" in st.session_state:
    result = st.session_state["latest_result"]
    result_json = json.dumps(result, ensure_ascii=False, indent=2)
    elapsed_ms = st.session_state["latest_elapsed_ms"]
    usage = st.session_state.get("latest_usage", {})

    st.header("Extraction result", icon=":material/description:")

    time_col, cost_col, token_col = st.columns(3)
    time_col.metric(
        "Elapsed time",
        f"{elapsed_ms / 1000:.1f} s",
        help="Time from submitting the form until the extraction service returned.",
    )
    estimated_cost = usage.get("estimated_provider_cost_usd")
    cost_display = (
        "$0.00"
        if usage.get("provider") == "ollama_local"
        else f"${estimated_cost:.6f}"
        if isinstance(estimated_cost, (int, float))
        else "Not available"
    )
    cost_col.metric(
        "Estimated provider cost",
        cost_display,
        help=(
            "This estimate uses token counts reported by Ollama for received "
            "responses, including retries. A timeout or transport failure may "
            "not report usage. Remote Ollama rates come from the optional "
            "MODEL_INPUT_COST_PER_1M_USD and MODEL_OUTPUT_COST_PER_1M_USD "
            "settings. Local electricity and hardware costs are not measured."
        ),
    )
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    token_display = (
        f"{prompt_tokens + completion_tokens:,}"
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
        else "Not reported"
    )
    token_col.metric(
        "Tokens across attempts",
        token_display,
        help="Prompt and completion tokens reported by Ollama, summed across retry responses.",
    )

    is_fallback = result.get("status") == "fallback"

    if is_fallback:
        st.warning(
            result["reason_for_review"],
            title="Extraction unavailable — human review required",
            icon=":material/person_search:",
        )
        st.info(
            "No extraction fields were invented. Review the original message "
            "and enter the details manually."
        )
    elif result["requires_human_review"]:
        st.warning(
            result["reason_for_review"] or "Please check this extraction.",
            title="Human review required",
            icon=":material/person_search:",
        )
    else:
        st.success(
            "The review rules did not flag this result.",
            title="No human review flagged",
            icon=":material/check_circle:",
        )

    if not is_fallback:
        with st.container(border=True):
            st.subheader("Contact details")
            name_col, contact_col = st.columns(2)
            name_col.metric("Name", result["name"] or "Not provided")
            contact_col.metric("Contact", result["contact"] or "Not provided")

        with st.container(border=True):
            st.subheader("Message classification")
            intent_col, urgency_col, budget_col = st.columns(3)
            intent_col.metric("Intent", result["intent"].replace("_", " ").title())
            urgency_col.metric("Urgency", result["urgency"].title())
            budget_col.metric(
                "Budget signal",
                result["budget_signal"].replace("_", " ").title(),
            )

    st.download_button(
        "Download JSON",
        data=result_json,
        file_name="extraction-result.json",
        mime="application/json",
        icon=":material/download:",
    )
    with st.expander("View raw JSON"):
        st.code(result_json, language="json")
