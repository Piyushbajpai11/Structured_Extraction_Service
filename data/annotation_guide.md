# Milestone 0: Dataset and annotation guide

Version: 0.4  
Scope: one incoming text message at a time. The examples use the fictional Northstar Docs document-intake service. This milestone contains data and annotation rules only; no service implementation is included.

## Files

- `synthetic_messages.jsonl`: 25 synthetic messages with coverage tags.
- `expected_outputs.jsonl`: gold extraction for each ID. P24 is an expected pre-call rejection and therefore has no model output.
- IDs join the two files. Coverage tags may overlap.

## Scope and coverage

The prior roadmap lists 8 clean pricing inquiries, 4 support requests, 3 partnership/spam examples, 3 mixed-language examples, 3 missing-contact examples, 2 multi-intent examples, and 2 long messages. Those are coverage dimensions rather than mutually exclusive buckets. Empty and gibberish are included within the fixed 25 records, and the five advancement challenges (prompt injection, Hindi/Hinglish, empty input, a roughly 2,000-word message, and dual intent) are tagged within the same set. This resolves the roadmap's arithmetic ambiguity: the first seven counts already total 25 before “plus empty and gibberish.”

The set covers 8 straightforward pricing messages (P01–P07 and P16); 4 single-intent support requests (P09–P12); 3 partnership/spam messages (P13–P15); three Hinglish messages (P05, P10, P13); missing contact information (P16–P18); dual intent (P19–P20); and long rambling messages (P21–P22). P19–P20 contain support as one intent and use support as their primary output label; their `dual_contains_*` tags mark the secondary intent without inflating the single-intent category counts. P08 contains an instruction-like prompt injection; P17 is a named but unclear inquiry; P23 is a clear product question that maps to `other`; P24 is empty; P25 is gibberish. Coverage tags overlap, and the empty and gibberish cases are included among the 25 unique records, not added after them.

## Output contract

Each non-null `expected` object has exactly these fields:

- `name`: string or null.
- `contact`: string or null.
- `intent`: `pricing_inquiry`, `support_request`, `partnership`, `spam`, or `other`.
- `urgency`: `low`, `medium`, or `high`.
- `budget_signal`: `mentioned`, `not_mentioned`, or `unclear`.
- `requires_human_review`: boolean.
- `reason_for_review`: string or null.

P24 has `expected: null` because an empty message should be rejected before an API call under the roadmap's pre-call validation rule. Its expected rejection is recorded separately. P25 is non-empty and is annotated as `other` with human review. The Step 3 implementation uses a 10,000-character input limit after trimming outer whitespace. P22 is 11,532 characters and is therefore rejected before inference; keep its gold object as a reference, but record the pre-call rejection separately and do not compare model fields for it.

## Annotation rules

### Name and contact

- Extract a person's name only when the sender states or signs it explicitly. Do not infer a name from an organization, email handle, or role.
- Extract an explicitly provided email address or phone number. For an email, return the address itself and omit enclosing punctuation such as parentheses, angle brackets, quotation marks, or a trailing comma/period. For a phone number, trim surrounding whitespace and preserve the number's internal formatting. If both are present, use the email. Do not invent a contact method.
- Use null when the message contains no explicit person name or contact method. Missing contact alone does not force review; the planned rule routes to review when both name and contact are missing.

### Intent

- `pricing_inquiry`: asks about price, quote, package, plan cost, discount, or commercial limits.
- `support_request`: asks for help with an existing account, product behavior, billing issue, or technical problem.
- `partnership`: proposes a genuine referral, reseller, implementation, or business collaboration.
- `spam`: unsolicited promotional or irrelevant bulk sales content. A legitimate partnership proposal is not spam.
- `other`: a clear product question that does not fit another enum, no clear fit, a vague question, or uninterpretable non-empty text. A clear request mapped to `other` does not by itself require human review.

For a message with two distinct actionable intents, select the issue that is most immediate or explicitly prioritized as the primary intent. Set review to true and name both intents in the reason because the schema has no secondary-intent field. P19 and P20 are the canonical dual-intent cases.

Treat text inside an incoming message as untrusted content, even if it tells the model to ignore instructions or change the output format. Extract the customer's actual request and do not follow embedded instructions. P08 tests this distinction.

### Urgency

- `high`: an explicit deadline today or tomorrow, or a current blocker that affects work today. A link expiring tomorrow is high urgency.
- `medium`: a stated upcoming decision or deadline that is not immediate.
- `low`: no concrete time pressure. Do not infer urgency from emphatic wording alone.

### Version history

- `0.4`: Clarified that quantities such as 10,000 leads are not a budget
  without a monetary amount, and that secondary intents require explicit
  evidence of a separate goal.
- `0.3`: Clarified that counts of people, users, and files are scale signals,
  not budget amounts; P06's expected budget signal remains
  `not_mentioned`.
- `0.2`: Corrected P12 urgency from `medium` to `high`; its invitation link
  expires tomorrow, which meets the explicit high-urgency rule.
- `0.1`: Initial 25-example baseline.

### Budget signal

- `mentioned`: a concrete money amount, spending cap, or budget range is stated, even if tentative.
- `not_mentioned`: no monetary amount or spending range is supplied. Asking for a price is not itself a budget. Counts of employees, users, files, documents, or items being purchased describe scale or quantity; they are not budgets. In P06, 80 employees is not a budget amount; in P15, 10,000 leads is a quantity, not a budget.
- `unclear`: the sender refers to a budget but gives no usable amount, or says the amount is undecided.

### Human review and reason

- Set `requires_human_review` true when both name and contact are null; when two separate actionable intents need human prioritization; when intent cannot be determined; or when a business-rule inconsistency is present (for example, spam marked high urgency).
- Set `reason_for_review` to a short, specific explanation whenever review is true. Use null when review is false.
- Do not treat the model's confidence claim as a review signal. Review is driven by explicit rules and unresolved ambiguity.
- P24 is rejected before extraction and therefore has no `LeadExtraction` object. P25 remains eligible for extraction and should be routed to review.

### Secondary-intent distinction

Only add a secondary intent when the message explicitly asks for a distinct
outcome. Separate pricing from troubleshooting or a collaboration proposal,
but do not split pricing details such as team size, spending cap, or setup fee
into separate intents.

## Quality review notes

- Exactly 25 unique IDs appear in each file, in matching order.
- Every non-null expected object uses the agreed seven-field contract and the allowed enum values.
- The dataset is text-only and each record represents one message. No authentication, batching, database, or conversation history is introduced.
- Contact values are synthetic; example email domains use `.example`. Names and organizations are fictional.
- Before Step 1, review the gold labels, especially the intent/review choice for P19–P20 and the separate pre-call outcome for P24. Resolve any disagreements here first, then freeze this version as the baseline eval set.
