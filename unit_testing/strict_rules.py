# unit_testing/strict_rules.py
from __future__ import annotations

import re
from typing import Dict, List, Optional


# -------------------------
# Canonical STRICT term sets
# -------------------------

ROLE_TERMS = [
    "dpo", "ciso", "cio", "cto",
    "legal counsel", "privacy team", "security team",
    "data owner", "data steward",
    "contractor", "third party",
]

ROLE_ALIASES: Dict[str, List[str]] = {
    "dpo": [r"\bdpo\b", r"data protection officer"],
    "ciso": [r"\bciso\b", r"chief information security officer"],
    "cio": [r"\bcio\b", r"chief information officer"],
    "cto": [r"\bcto\b", r"chief technology officer"],
    "legal counsel": [r"\blegal counsel\b", r"\battorney\b", r"\blawyer\b"],
    "privacy team": [r"\bprivacy team\b"],
    "security team": [
        r"\bsecurity team\b",
        r"\binformation security\b",
        r"\binfosec\b",
        r"\binfo\s*sec\b",
        r"\biso\b",
        r"\bsecurity officer\b",
        r"\binformation security officer\b",
    ],
    "data owner": [r"\bdata owner\b"],
    "data steward": [r"\bdata steward\b"],
    "contractor": [r"\bcontractor(s)?\b"],
    # Handle plural forms cleanly
    "third party": [r"\bthird[- ]part(y|ies)\b"],
}

DATA_CLASS_TERMS = [
    "pii",
    "personal data",
    "personally identifiable information",
    "phi",
    "health data",
    "financial data",
    "payment card data",
    "pci",
    "biometric data",
    "credentials",
    "passwords",
    "api keys",
]

DATA_CLASS_ALIASES: Dict[str, List[str]] = {
    "pii": [r"\bpii\b", r"personally identifiable information", r"personal information"],
    "personal data": [r"personal data", r"personal information"],
    "personally identifiable information": [r"personally identifiable information", r"\bpii\b"],
    "phi": [r"\bphi\b", r"protected health information"],
    "health data": [r"health data", r"medical data", r"patient data"],
    "financial data": [r"financial data", r"bank account", r"routing number"],
    "payment card data": [r"payment card", r"card number", r"cvv"],
    "pci": [r"\bpci\b", r"payment card"],
    "biometric data": [r"biometric", r"fingerprint", r"faceprint"],
    "credentials": [r"credentials", r"login", r"username"],
    "passwords": [r"password(s)?"],
    "api keys": [r"api key(s)?", r"access token(s)?", r"secret key(s)?"],
}

CADENCE_TERMS = [
    "daily", "weekly", "monthly", "quarterly", "annually",
    "within 24 hours", "within 72 hours", "within 30 days",
]

LEGAL_CLAIM_TERMS = [
    "reportable",
    "breach notification",
    "notify regulators",
    "notify the regulator",
    "notify authorities",
    "regulatory notification",
    "regulatory reporting",
    "report to regulators",
    "report to the regulator",
    "report to authorities",
    "law enforcement",
    "attorney general",
    "required by law",
    "legal requirement",
    "regulatory requirement",
    "gdpr",
    "ccpa",
    "hipaa",
]

LEGAL_TERMS = [
    "gdpr", "ccpa", "hipaa", "sox", "glba",
    "regulator", "supervisory authority", "regulatory notification",
    "breach notification", "notify affected individuals",
]

PII_EVIDENCE_PATTERNS = [
    r"\bname(s)?\b",
    r"\bphone number(s)?\b|\btelephone\b|\bmobile\b",
    r"\bemail\b|\be-mail\b",
    r"\baddress\b",
    r"\bip address\b|\bip\b",
    r"\bssn\b|\bsocial security\b",
    r"\bpassport\b|\bdriver'?s license\b",
]


# -------------------------
# Helpers
# -------------------------

def raw_supports_pii(raw: str) -> bool:
    raw_l = (raw or "").lower()
    return any(re.search(p, raw_l) for p in PII_EVIDENCE_PATTERNS)


def _present(text_lower: str, canonical: str, aliases: Optional[Dict[str, List[str]]] = None) -> bool:
    if aliases and canonical in aliases:
        return any(re.search(pat, text_lower) for pat in aliases[canonical])
    return canonical in text_lower


def diff_terms(output: str, raw: str, terms: List[str], aliases: Optional[Dict[str, List[str]]] = None) -> List[str]:
    out_l = (output or "").lower()
    raw_l = (raw or "").lower()
    introduced: List[str] = []
    for term in terms:
        if _present(out_l, term, aliases) and not _present(raw_l, term, aliases):
            introduced.append(term)
    return introduced


NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def diff_numbers(output: str, raw: str) -> List[str]:
    raw_nums = set(NUM_RE.findall(raw or ""))
    out_nums = NUM_RE.findall(output or "")

    # Ignore list indices like "1." or "2)"
    ignored = set()
    for m in re.finditer(r"(?m)^\s*(\d{1,2})\s*[\.)]\s+", output or ""):
        ignored.add(m.group(1))
    for m in re.finditer(r"(?i)\bstep\s+(\d{1,2})\b", output or ""):
        ignored.add(m.group(1))

    bad: List[str] = []
    for n in out_nums:
        if n in raw_nums or n in ignored:
            continue
        bad.append(n)

    # de-dupe preserving order
    seen = set()
    out = []
    for n in bad:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def compute_flags(output: str, raw: str) -> Dict[str, List[str]]:
    roles = diff_terms(output, raw, ROLE_TERMS, ROLE_ALIASES)
    data_classes = diff_terms(output, raw, DATA_CLASS_TERMS, DATA_CLASS_ALIASES)

    # If raw includes concrete PII evidence, don't flag generic PII mentions as "introduced"
    if raw_supports_pii(raw):
        data_classes = [t for t in data_classes if t not in ("pii", "personal data", "personally identifiable information")]

    cadence = diff_terms(output, raw, CADENCE_TERMS)
    legal_terms = list(dict.fromkeys(LEGAL_CLAIM_TERMS + LEGAL_TERMS))
    legal = diff_terms(output, raw, legal_terms)

    numbers = diff_numbers(output, raw)

    return {
        "numbers_not_in_input": numbers,
        "roles_not_in_input": roles,
        "data_classes_not_in_input": data_classes,
        "cadence_not_in_input": cadence,
        "legal_claims_not_in_input": legal,
    }


def any_flagged(flags: Dict[str, List[str]]) -> bool:
    return any(bool(v) for v in flags.values())


# -------------------------
# Deterministic scrubber
# -------------------------

def scrub_introduced_roles(text: str, introduced_roles: List[str]) -> str:
    """
    Deterministic post-processor used ONLY when verifier flags invented roles.
    Rewrites invented role mentions to a neutral placeholder that won't trip the gate.
    """
    if not introduced_roles:
        return text

    t = text or ""
    repl = "designated responsible owner (TBD)"

    patterns: List[str] = []
    for role in introduced_roles:
        patterns += ROLE_ALIASES.get(role, [])
        patterns += [re.escape(role)]

    for pat in patterns:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)

    t = re.sub(r"\s{2,}", " ", t)
    return t
