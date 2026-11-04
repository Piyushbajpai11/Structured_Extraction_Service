# Failure register

This register records how the local service can fail, how it responds, and
what remains uncertain. It is a project learning artifact, not a claim that
every possible failure has been eliminated.

| ID | Failure mode | Effect | Current handling | Remaining risk |
|---|---|---|---|---|
| F-01 | Empty or too-short message | No useful content can be extracted. | Python rejects it before inference and records a rejected usage event. | The user must correct the message. |
| F-02 | Message exceeds the configured limit | A very large request could consume time or memory. | Python rejects it before inference; the default maximum is 4,000 characters. | The user may need to shorten a legitimate message. |
| F-03 | Ollama unavailable or request times out | No model result arrives in time. | One retry is allowed for retryable failures; after that, a typed fallback requests human review and fabricates no fields. | A person must enter details manually; a timed-out remote server may not return billable token counts. |
| F-04 | Model returns malformed JSON or wrong field types | The response cannot satisfy the service contract. | Pydantic validates the response; one retry is allowed, followed by the safe fallback. | Different Ollama/model versions may behave differently. |
| F-05 | Valid-shaped but factually wrong extraction | Schema validation succeeds even though a value is wrong. | Synthetic expected outputs, field metrics, failure reports, and human review support detection. | Small synthetic evaluation results do not guarantee real-world accuracy. |
| F-06 | Ambiguous request or multiple separate goals | The service may misclassify intent or miss a need for review. | Prompt guidance asks the model to identify unclear and additional intents; Python applies deterministic review rules. | Model judgment can still be inconsistent; inspect per-example failures. |
| F-07 | Instructions embedded in the customer message | Untrusted content may try to change extraction behavior. | System instructions mark the message as untrusted; schema validation restricts output shape; P08 covers an injection-style example. | Prompt instructions and JSON constraints are not a proof against every attack. |
| F-08 | Contact formatting differs from the expected label | Literal comparison can report a mismatch for equivalent formatting. | Evaluation reports exact comparison and a separate normalized email/phone comparison. | Normalization is deliberately limited and may not handle every international phone convention. |
| F-09 | Usage log cannot be written | A request succeeds but usage reporting is incomplete. | The service warns on stderr and continues to return the extraction. | Cost and latency summaries are incomplete until logging works. |
| F-10 | Missing remote model pricing or token usage | A remote provider estimate cannot be calculated reliably. | Both optional per-million-token rates are required; missing rate or token data produces an unavailable estimate. | Estimates may differ from provider invoices; transport failures may omit usage metadata. |

## How to use this register

When a new failure appears, add its observed condition, effect, current
handling, and remaining risk. Link it to a synthetic example or automated
check when one exists. Do not mark a risk as solved solely because one sample
worked once.
