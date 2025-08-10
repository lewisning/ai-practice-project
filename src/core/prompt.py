from typing import List, Dict, Any

SYSTEM = """You are a Tucows Domains Knowledge Assistant for support agents.
            Answer ONLY with grounded facts from the provided CONTEXT.
            If the context is insufficient or conflicting, say so and propose the minimal next step.
            Return a single valid JSON object that matches the OUTPUT SCHEMA. No extra text.
         """

SCHEMA_HINT = """OUTPUT SCHEMA:
                {
                  "answer": string,
                  "references": string[],  // use "title · section · id"
                  "action_required": "no_escalation_needed" | "request_more_information" | "escalate_to_abuse_team" | "escalate_to_billing_team" | "escalate_to_support_level_2"
                }
              """

def build_user_prompt(ticket_text: str, snippets: List[Dict[str, Any]]) -> str:
    lines = []
    lines.append("TICKET:\n" + ticket_text.strip())
    lines.append("\nCONTEXT:")
    for i, sn in enumerate(snippets, 1):
        lines.append(f"[SNIPPET {i}]")
        lines.append(f"- id: {sn['id']}")
        lines.append(f"- title: {sn['payload'].get('doc','')}")
        lines.append(f"- section: {sn['payload'].get('section','')}")
        lines.append(f'- text: "{sn["payload"].get("text","").strip()}"\n')
    lines.append("INSTRUCTIONS:")
    lines.append("1) Use only CONTEXT to answer.")
    lines.append('2) Cite the most relevant 2–5 snippets in "references" using "title · section · id".')
    lines.append("3) If steps depend on user verification/ownership, ask for it and choose action_required accordingly.")
    lines.append("   If the ticket mentions a domain being suspended/suspension or policy enforcement, set action_required to escalate_to_abuse_team.")
    lines.append("4) Output MUST be a single JSON object and match OUTPUT SCHEMA exactly.")
    return "\n".join(lines)

def output_schema_hint() -> str:
    return SCHEMA_HINT
