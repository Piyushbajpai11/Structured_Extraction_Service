# Two-minute project demo

## Before recording

- Open PowerShell in the project folder and activate `.venv`.
- Start Ollama and confirm `qwen3.5:4b` is installed.
- Start Streamlit and open the local page.
- Use synthetic examples only; do not show API keys or `.env` contents.

## Walkthrough

| Time | Show and say |
|---|---|
| 0:00–0:15 | Introduce the goal: convert one free-form message into a validated JSON record, with a person reviewing uncertain cases. Mention that the model runs locally. |
| 0:15–0:35 | Show the message form and explain that Python checks the input before inference. Paste a clear synthetic pricing example from `data/synthetic_messages.jsonl`. |
| 0:35–1:00 | Submit it. Point out the extracted contact, intent, urgency, budget signal, review decision, token count, elapsed time, and local provider cost of $0. |
| 1:00–1:20 | Show the downloadable JSON and explain that both the UI and command-line app call the same extraction service. |
| 1:20–1:40 | Submit a synthetic ambiguous or multi-intent example. Explain why the service flags it for human review instead of treating valid JSON as automatically correct. |
| 1:40–2:00 | Show the JSON/Markdown/CSV reports from a completed evaluation, or run `python evaluate.py --ids P15` if the model is warm. State the v0.4 baseline and its limitation: 25 synthetic examples are a learning evaluation, not proof of real-world performance. |

If the first model response takes longer than the remaining demo time, prepare
the model before recording and rehearse with a single example. Do not hide a
failed response; explain the safe fallback if it occurs.
