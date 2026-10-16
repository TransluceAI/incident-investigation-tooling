"""Pass 0: cheap, non-LLM triggers that send a window straight to the reasoning pass.

These cover patterns the first pass under-flagged in the Arquivo comparison:
backend ports, request code that overrides the Host header or TLS server name,
file-version paths, and CAPTCHA-to-OCR chains. A hit is a reason to look again,
never a label.

A full-corpus scan (2026-10-16) found every true positive in request or CDX
URLs and their decoded views or programs, while body text produced the noise
(JavaScript localhost ports, wiki links to :8080, Suricata "TLS SNI" alerts,
reCAPTCHA site keys, urlscan hostname lists). So the rules read only the
window's non-body segments, except the precise dav/versions path. They are not
yet measured against labelled data.
"""
import re

from .shapes import BULK_KINDS

URL_PORT = re.compile(r"(?i)\bhttps?://[a-z0-9.\-\[\]]+:(\d{2,5})(?!\d)")
DEFAULT_PORTS = {"80", "443"}
HOST_OVERRIDE = re.compile(r"(?i)headers[\"']?\s*:\s*\{\s*[\"']?host[\"']?\s*:|\bservername[\"']?\s*:")
VERSION_PATH = re.compile(r"(?i)\bdav/versions\b")
CAPTCHA_ENDPOINT = re.compile(r"(?i)captcha")
OCR_API = re.compile(r"(?i)ocr\.space|tesseract")


def context_segments(window):
    """Text of each non-body segment of a prepared window (URLs, decodings, programs, provenance)."""
    text = window["text"]
    return [text[a:b] for sid, (a, b) in window["segment_spans"].items()
            if window["segment_kinds"].get(sid) not in BULK_KINDS]


def hits(window) -> list:
    """Names of the pass-0 rules that fire on one prepared window."""
    context = context_segments(window)
    found = []
    if any(port not in DEFAULT_PORTS for segment in context for port in URL_PORT.findall(segment)):
        found.append("backend_port")
    if any(HOST_OVERRIDE.search(segment) for segment in context):
        found.append("host_override")
    if VERSION_PATH.search(window["text"]):
        found.append("file_version_path")
    if any(CAPTCHA_ENDPOINT.search(segment) and OCR_API.search(segment) for segment in context):
        found.append("captcha_ocr_chain")
    return found
