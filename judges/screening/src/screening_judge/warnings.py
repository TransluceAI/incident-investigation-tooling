"""Review hints, never verdicts. Name recognition and setup detection are heuristics."""
import re

# Unrecognized names are queued for review, not declared false attribution.
MODEL_OR_PROVIDER = re.compile(
    r'\b(openai|gpt[\w.-]*|chatgpt|o[134](?:[- ]|\b)|anthropic|claude|'
    r'google|gemini|deepmind|meta|llama|mistral|qwen|alibaba|deepseek|xai|grok|'
    r'cohere|muse spark|command[- ]r|moonshot|kimi|gemma|phi[- ]?\d)\b', re.I)
SETUP = re.compile(r'\b(config(?:uration)?|initializ\w*|log(?:ging)? (?:level|file)|'
                   r'connect[_ -]?timeout|read[_ -]?timeout|user[_ -]?agent)\b', re.I)
REQUEST = re.compile(r'(?:\b(?:GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+\S+\s+HTTP/\d|'
                     r'\b(?:sending|sent|received|response status|status_code)\b)', re.I)

# A response status stated as observed ("returned 403", "200 OK", "403 PermissionDeniedException").
# Judges have asserted codes the cited text never showed; a URL path like /status/204 is not a claim.
STATUS_CLAIM = re.compile(
    r"(?i)\b(?:status(?:\s+code)?|returned|returns|returning|responded|responds|response|answered|"
    r"HTTP(?:/[\d.]+)?)(?:\s+(?:an?|the|with(?:\s+an?)?))?[\s:=(]{1,3}([1-5]\d\d)\b"
    r"|\b([1-5]\d\d)\)?\s*(?:status|response|error|ok\b|forbidden|unauthorized|not found|bad request|"
    r"bad gateway|server error|unavailable|no content|too many|\w*exception|\w*error)")


def _status_not_cited(result, evidence):
    actions = result.get("actions")
    claims = [result.get(k) for k in ("rationale", "boundary", "authorization")]
    claims += [a.get(k) for a in (actions if isinstance(actions, list) else []) if isinstance(a, dict)
               for k in ("target", "action")]
    # Consecutive chunks of one segment can split a token; any other pair stays apart.
    cited, prev = "", None
    for e in sorted(evidence, key=lambda e: (e["segment_id"], e["chunk_id"])):
        cited += ("" if prev == (e["segment_id"], e["chunk_id"] - 1) else "\n") + e["quote"]
        prev = (e["segment_id"], e["chunk_id"])
    return any(not re.search(rf"(?<!\d){code}(?!\d)", cited)
               for claim in claims if isinstance(claim, str)
               for m in STATUS_CLAIM.finditer(claim) for code in m.groups() if code)


def review_warnings(result, evidence, confidence_semantics="behaviour_support"):
    if not isinstance(result, dict):
        return []
    flags = []
    confidence = result.get("confidence")
    if (confidence_semantics == "behaviour_support" and result.get("label") == "no"
            and type(confidence) in (int, float) and confidence > .5):
        flags.append("legacy_confidence_polarity")
    if result.get("provider_attribution", "not_established") != "not_established":
        if not MODEL_OR_PROVIDER.search(str(result.get("claimed_actor") or "")):
            flags.append("provider_attribution_needs_named_model_review")
    if "attempt_present" in result and ((result.get("label") == "no" and result["attempt_present"])
                                        or (result.get("label") == "yes" and not result["attempt_present"])):
        flags.append("attempt_label_contradiction")
    actions = result.get("actions")
    for action in actions if isinstance(actions, list) else []:
        if not isinstance(action, dict):
            continue
        if action.get("observed_stage") != "recorded_request":
            continue
        if "evidence" in action:
            keys = {(e["segment_id"], e["chunk_id"]) for e in (action["evidence"] or []) if isinstance(e, dict) and "segment_id" in e and "chunk_id" in e}
            cited = [e for e in evidence if (e["segment_id"], e["chunk_id"]) in keys]
        else:
            cited = [e for e in evidence if e["segment_id"] in action.get("evidence_segment_ids", [])]
        text = "\n".join(e["quote"] for e in cited)
        if SETUP.search(text) and not REQUEST.search(text):
            flags.append("recorded_request_cites_setup_review")
    if evidence and _status_not_cited(result, evidence):
        flags.append("status_not_in_cited_text_review")
    return sorted(set(flags))
