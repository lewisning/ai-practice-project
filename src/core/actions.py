import re
from typing import Literal, List, Dict, Any

Action = Literal[
    "no_escalation_needed",
    "request_more_information",
    "escalate_to_abuse_team",
    "escalate_to_billing_team",
    "escalate_to_support_level_2",
]

_ABUSE = re.compile(r"\b(abuse|malware|phishing|spam|fraud)\b", re.I)
_BILL  = re.compile(r"\b(billing|refund|chargeback|invoice)\b", re.I)
_IDV   = re.compile(r"\b(verification|ownership|identity|id check)\b", re.I)
_SUSP  = re.compile(r"\b(suspend|suspension|suspended)\b", re.I)

def enforce_action(ticket_text: str, snippets: List[Dict[str, Any]], llm_action: Action) -> Action:
    text = ticket_text.lower()

    # Rule 1: If the ticket text contains Suspension/Abuse keywords, enforce escalation to abuse team
    if _SUSP.search(text) or _ABUSE.search(text):
        return "escalate_to_abuse_team"

    # Rule 2: If any retrieved snippet has "suspension" or "abuse" in the title, enforce escalation to abuse team
    for sn in snippets:
        title = (sn.get("payload") or {}).get("doc", "").lower()
        if "suspension" in title or "abuse" in title:
            return "escalate_to_abuse_team"

    # Rule 3: Billing-related keywords → Escalate to billing team
    if _BILL.search(text):
        return "escalate_to_billing_team"

    # Rule 4: Identity/Verification keywords then escalate to support level 2
    if _IDV.search(text):
        return "escalate_to_support_level_2"

    # Rule 5: If the LLM suggests "no_escalation_needed" but the ticket text contains "verification" or "ownership", override to "escalate_to_support_level_2"
    return llm_action or "no_escalation_needed"
