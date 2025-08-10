import json, orjson
from typing import Dict, Any, List
from openai import OpenAI
import os
from dotenv import load_dotenv

from src.rag.merged_retriever import search_merged
from src.core.prompt import SYSTEM, build_user_prompt, output_schema_hint
from src.api.schemas import TicketResponse
from src.core.actions import enforce_action

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _pick_snippets(ticket_text: str, top_k: int = 8) -> List[Dict[str, Any]]:
    filters = {} # e.g., {"product": "domains", "lang": "en"}
    return search_merged(ticket_text, top_k=top_k, filters=filters or None, alpha=0.7)

def _call_llm(messages: List[Dict[str, str]]) -> str:
    # Response format: strict JSON object
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_GPT_NAME"),
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=messages
    )
    return resp.choices[0].message.content

def resolve_ticket(ticket_text: str, top_k: int = 8) -> TicketResponse:
    # 1. Retrieval
    snippets = _pick_snippets(ticket_text, top_k=top_k)

    # 2. Construct prompt
    user_prompt = build_user_prompt(ticket_text, snippets)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "assistant", "content": output_schema_hint()},
        {"role": "user", "content": user_prompt},
    ]

    # 3. LLM Call
    raw = _call_llm(messages)

    # 4. Parse and validate JSON
    def _parse(s: str):
        try:
            return TicketResponse.model_validate_json(s)
        except Exception:
            fix_msg = messages + [
                {"role": "system", "content": "Your previous output was not valid JSON per the schema. Reply again with ONLY the JSON object."}
            ]
            fixed = _call_llm(fix_msg)
            return TicketResponse.model_validate_json(fixed)

    resp: TicketResponse = _parse(raw)

    # 5. Reference filtering, ensure IDs exist, fallback to first two snippets
    allowed_ids = set(sn["id"] for sn in snippets)
    filtered_refs = []
    for r in resp.references:
        # Expect format: "title · section · id"
        parts = r.rsplit("·", 1)
        if len(parts) == 2 and parts[1].strip() in allowed_ids:
            filtered_refs.append(r)
    if not filtered_refs:
        # Auto-fallback to first two snippets if none valid
        def fmt(sn):
            return f"{sn['payload'].get('doc','')} · {sn['payload'].get('section','')} · {sn['id']}"
        filtered_refs = [fmt(sn) for sn in snippets[:2]]

    # 6. Enforce action rules
    final_action = enforce_action(ticket_text, snippets, resp.action_required)

    # 7. Return final response
    return TicketResponse(
        answer=resp.answer.strip(),
        references=filtered_refs,
        action_required=final_action
    )
