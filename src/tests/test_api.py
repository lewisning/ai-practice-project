import re
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_ingest_and_search_roundtrip():
    # Ingest a chunk
    payload = {
        "doc": "Policy: Domain Suspension Guidelines",
        "section": "4.2",
        "anchor_id": "para-17",
        "text": "A domain may be suspended due to invalid WHOIS information. To reactivate, update WHOIS and provide proof of registrant identity.",
        "product": "domains",
        "lang": "en",
    }
    r = client.post("/ingest", json=payload)
    assert r.status_code == 200 and r.json()["ok"] is True
    pid = r.json()["id"]

    # Search for the ingested chunk
    q = {
        "query": "reactivate a suspended domain due to invalid WHOIS",
        "top_k": 3,
        "product": "domains",
        "lang": "en",
    }
    r = client.post("/search_merged", json=q)
    assert r.status_code == 200
    hits = r.json()["hits"]
    assert any(h["id"] == pid for h in hits), "merged search should recall the ingested chunk"

def test_resolve_ticket_schema_and_action():
    req = {
        "ticket_text": "My domain was suspended and I didn’t get any notice. How can I reactivate it?",
        "top_k": 8,
    }
    r = client.post("/resolve-ticket", json=req)
    assert r.status_code == 200
    data = r.json()

    # Schema validation
    assert isinstance(data["answer"], str) and len(data["answer"]) > 10
    assert isinstance(data["references"], list) and len(data["references"]) >= 1
    assert data["action_required"] in {
        "no_escalation_needed",
        "request_more_information",
        "escalate_to_abuse_team",
        "escalate_to_billing_team",
        "escalate_to_support_level_2",
    }

    # Action validation
    assert data["action_required"] == "escalate_to_abuse_team"

    # References validation, should be in "title · section · uuid" format
    tail_ok = False
    for ref in data["references"]:
        # Split on the last "·" to check the tail
        parts = ref.rsplit("·", 1)
        if len(parts) == 2 and parts[1].strip():
            tail_ok = True
            break
    assert tail_ok, "references should end with a valid id in 'title · section · id' format"
