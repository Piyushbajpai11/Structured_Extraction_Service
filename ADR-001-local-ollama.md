# ADR-001: Run the first version with local Ollama

- **Status:** Accepted
- **Date:** 2026-11-04

## Context

This is a first AI engineering project. The goal is to learn a complete
structured-extraction workflow while keeping the first working version local.
Hosted inference can require billing, account credits, API credentials, and a
network-accessible service. A local Ollama model is already available on the
development computer.

## Decision

Use Ollama with the locally installed `qwen3.5:4b` model for inference. Keep
input validation, Pydantic response validation, deterministic review rules,
bounded retries, safe fallback behavior, and usage logging in Python. Expose
the same extraction service through the command line and Streamlit interface.

Do not deploy a public version until a hosted inference endpoint, cost plan,
and secret-handling approach have been chosen.

## Alternatives considered

1. **Hosted inference API:** useful for sharing a live app, but requires a
   reachable endpoint and provider billing or credits. Deferred.
2. **Rules-only extraction:** avoids model setup, but would not teach the
   project goal of schema-constrained LLM extraction.
3. **Fine-tuning a model:** unnecessary for this small labelled dataset and
   beyond the first project's learning goals.

## Consequences

- No hosted model API key is required for the current local setup.
- The provider charge for local inference is $0; electricity and hardware
  costs are not measured.
- Response time depends on the computer and installed model.
- A syntactically valid JSON object can still contain incorrect information;
  evaluation and human review remain necessary.
- Ollama's structured-output behavior can vary by model and version, so live
  compatibility must be checked when either changes.
- Localhost is not reachable by a public hosting service. Deployment requires
  a separate hosted-inference decision.

## Revisit this decision when

- A public URL is required.
- A hosted provider and its current rates have been selected.
- Data-handling terms and secret storage have been reviewed.
- The provider adapter passes the same schema, validation, fallback, and
  evaluation checks.
