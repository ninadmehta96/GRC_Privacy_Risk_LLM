# unit_testing/test_strict_rules.py
from unit_testing.strict_rules import compute_flags, scrub_introduced_roles


def test_third_party_plural_alias_not_flagged():
    raw = "We share data with third parties under contract."
    out = "We share data with a third party under contract."
    flags = compute_flags(out, raw)
    assert "third party" not in flags["roles_not_in_input"]


def test_security_team_is_flagged_when_not_in_raw():
    raw = "This policy defines incident response steps."
    out = "The security team shall review incidents."
    flags = compute_flags(out, raw)
    assert "security team" in flags["roles_not_in_input"]


def test_scrubber_removes_security_team_aliases():
    txt = "The Information Security Officer (ISO) and security team will approve exceptions."
    cleaned = scrub_introduced_roles(txt, ["security team"])
    assert "security team" not in cleaned.lower()
    assert "information security" not in cleaned.lower()
    assert "iso" not in cleaned.lower()
    assert "designated responsible owner" in cleaned.lower()
