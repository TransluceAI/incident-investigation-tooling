"""Label scores from the shared runner's final-answer logprobs.

The runner returns raw chat-completions style logprobs; turning them into label
scores is this judge's job. Scores are diagnostic first-token branch scores,
NOT calibrated label probabilities.
"""
import json
import math
import re

from .schemas import LABELS

LABEL_KEY = re.compile(r'"label"\s*:\s*"$')


def label_probabilities(logprobs):
    """Branch scores at the top-level label value, or None.

    Require an unambiguous label boundary and candidates for every label. A
    missing candidate is unknown, never zero. Multi-token labels are represented
    by their unique first prefix only; no sequence probability is claimed.
    Combined boundary/value tokens conservatively produce no score.
    """
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not content:
        return None
    text = ""
    for entry in content:
        # The key pattern ends in a quote; skip the regex for prefixes that cannot match.
        if text.endswith(('"', '"\n')) and LABEL_KEY.search(text):
            # Parse the prefix to ensure this is the actual top-level field,
            # rather than a quoted label-looking string inside the rationale.
            try:
                prefix = json.loads(text + '__score_boundary__"}')
            except (ValueError, TypeError):
                return None
            if not isinstance(prefix, dict) or prefix.get("label") != "__score_boundary__":
                return None
            mass, found = dict.fromkeys(LABELS, 0.0), set()
            for candidate in entry.get("top_logprobs") or ():
                piece = (candidate.get("token") or "").rstrip('" ,}')
                matches = [label for label in LABELS if piece and label.startswith(piece)]
                if len(matches) == 1:
                    lp = candidate.get("logprob")
                    if type(lp) not in (int, float) or not math.isfinite(lp) or lp > 0:
                        return None
                    mass[matches[0]] += math.exp(lp)
                    found.add(matches[0])
            if found != set(LABELS):
                return None
            total = sum(mass.values())
            return {k: v / total for k, v in mass.items()} if total > 0 else None
        token = entry.get("token")
        if not isinstance(token, str):
            return None
        text += token
    return None
