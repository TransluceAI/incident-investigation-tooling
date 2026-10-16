"""Per-axis output schemas.

Property order is generation order under grammar-constrained decoding, so the
model writes its cited evidence and reasoning before the label (all surveyed
frameworks and vendor guidance put reasoning first). Labels are binary plus an
abstain value; likely/possible are derived later from calibrated probabilities.

Only JSON-schema keywords already exercised with the cluster's vLLM grammar
backend are used (type, enum, properties, required, additionalProperties,
items, maxItems, maxLength, minimum). Non-empty and uniqueness rules are
enforced by validate.py instead.
"""
AXES = ("hacking", "agentic_activity", "coordination")
LABELS = ("yes", "no", "insufficient_evidence")
HACKING_CATEGORIES = ("sql_injection", "xss", "command_injection", "template_injection",
                      "path_traversal_or_file_inclusion", "source_or_config_disclosure_probe",
                      "ssrf", "account_or_credential_abuse", "bot_or_access_control_bypass",
                      "flooding_or_dos", "other")
# Findings submission form choices for each category (the form's own wording).
FINDINGS_CYBERATTACK = {
    "sql_injection": "SQL injection", "xss": "Cross-site scripting (XSS)",
    "command_injection": "Command injection", "template_injection": "Template injection",
    "path_traversal_or_file_inclusion": "Path traversal / file inclusion",
    "ssrf": "Server-side request forgery (SSRF)",
    "account_or_credential_abuse": "Account or credential abuse",
    "bot_or_access_control_bypass": "Bypassing bot protection or access controls",
    "flooding_or_dos": "Flooding / denial of service",
    "source_or_config_disclosure_probe": "Other: source/configuration disclosure probing",
    "other": "Other",
}
STAGES = ("discussion_or_planning", "published_payload", "recorded_request",
          "observed_response", "demonstrated_outcome")
RESPONDERS = ("archive_replay", "intermediary_service", "ultimate_target", "unknown")
OUTCOMES = ("supported_success", "blocked_or_failed", "unknown", "not_applicable")
MECHANISMS = ("shared_results", "relayed_message", "shared_state", "addressed_to_agents",
              "other", "none")


def _text(n):
    return {"type": "string", "maxLength": n}


EVIDENCE = {
    "type": "array", "maxItems": 4,
    "items": {"type": "object", "additionalProperties": False,
              "required": ["segment_id", "chunk_id", "location"],
              "properties": {"segment_id": _text(64),
                             "chunk_id": {"type": "integer", "minimum": 1},
                             "location": _text(160)}},
}
ACTION = {
    "type": "object", "additionalProperties": False,
    "required": ["target", "action", "observed_stage", "responding_system", "outcome",
                 "evidence_segment_ids"],
    "properties": {"target": _text(300), "action": _text(300),
                   "observed_stage": {"enum": list(STAGES)},
                   "responding_system": {"enum": list(RESPONDERS)},
                   "outcome": {"enum": list(OUTCOMES)},
                   "evidence_segment_ids": {"type": "array", "maxItems": 4, "items": _text(64)}},
}


def schema(axis: str) -> dict:
    if axis not in AXES:
        raise ValueError(f"Unknown axis: {axis}")
    props = {"evidence": EVIDENCE, "policy_basis": _text(160)}
    if axis == "hacking":
        props |= {"categories": {"type": "array", "maxItems": 4,
                                 "items": {"enum": list(HACKING_CATEGORIES)}},
                  "boundary": _text(300), "authorization": _text(300),
                  "actions": {"type": "array", "maxItems": 4, "items": ACTION}}
    elif axis == "agentic_activity":
        props |= {"claimed_actor": _text(200),
                  "provider_attribution": {"enum": ["not_established", "claimed", "corroborated"]}}
    else:
        props |= {"mechanism": {"enum": list(MECHANISMS)}}
    props |= {"rationale": _text(600), "label": {"enum": list(LABELS)},
              "injection_detected": {"type": "boolean"},
              "missing_evidence": {"type": "array", "maxItems": 4, "items": _text(200)}}
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object",
            "additionalProperties": False, "required": list(props), "properties": props}
