"""Pass 0: cheap, non-LLM triggers that send a window straight to the reasoning pass.

These cover patterns the first pass under-flagged in the Arquivo comparison:
backend ports, requests to a raw IP address with an overridden Host header or
SNI, file-version paths, and CAPTCHA-to-OCR chains. A hit is a reason to look
again, never a label. The patterns are deliberately narrow and not yet measured
against labelled data.
"""
import re

URL_PORT = re.compile(r"(?i)\bhttps?://[a-z0-9.-]+:(\d{2,5})(?=[/?#\s\"'<>]|$)")
DEFAULT_PORTS = {"80", "443"}
IP_URL = re.compile(r"(?i)\bhttps?://(?:\d{1,3}\.){3}\d{1,3}(?::\d{2,5})?(?=[/?#\s\"'<>]|$)")
HOST_OVERRIDE = re.compile(r"(?im)(?:^|[\s\"'{,;])(?:host|sni|servername|server_name)[\"']?\s*[:=]")
VERSION_PATH = re.compile(r"(?i)\bdav/versions\b")
CAPTCHA = re.compile(r"(?i)captcha")
OCR = re.compile(r"(?i)\bocr\b|tesseract|ocr\.space|image[-_ ]?to[-_ ]?text")


def hits(text: str) -> list:
    """Names of the pass-0 rules that fire on one window's text."""
    found = []
    if any(port not in DEFAULT_PORTS for port in URL_PORT.findall(text)):
        found.append("backend_port")
    if IP_URL.search(text) and HOST_OVERRIDE.search(text):
        found.append("direct_ip_host_override")
    if VERSION_PATH.search(text):
        found.append("file_version_path")
    if CAPTCHA.search(text) and OCR.search(text):
        found.append("captcha_ocr_chain")
    return found
