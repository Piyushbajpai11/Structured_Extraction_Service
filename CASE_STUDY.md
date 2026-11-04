# Case study: structured extraction from customer messages

1. **Problem:** Turn free-form customer messages into consistent fields and route ambiguous cases for human review.
2. **Approach:** Build a local Python service using Pydantic, Ollama with Qwen 3.5 4B, deterministic review rules, bounded retries, and a Streamlit interface.
3. **Evaluation data:** Create 25 labelled synthetic examples; 23 were eligible for inference and two were correctly rejected before calling the model.
4. **Recorded v0.4 result:** All 23 model-eligible examples matched the expected extraction and routing fields when review-reason wording was ignored; classification macro/micro F1 was 1.00 and normalized contact matched 23/23. Literal full-record match was 18/23 (78.3%) because five review explanations used different wording.
5. **Limit:** This small synthetic set does not establish real-customer performance. Live provider billing, public deployment, and measured local energy costs remain outside the current baseline.
