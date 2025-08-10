import os, uuid, re, time
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as pdf_extract_text

from src.rag.embedding import embed_texts
from src.rag.qdrant_store import qdrant_store
from src.utils.settings import settings

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_WS = re.compile(r"\s+")

def _read_text_from_file(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in [".md", ".txt"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suf in [".html", ".htm"]:
        html = path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "lxml")
        # keep visible text only
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        return _WS.sub(" ", soup.get_text("\n")).strip()
    if suf == ".pdf":
        return pdf_extract_text(str(path)) or ""
    raise ValueError(f"Unsupported file type: {suf}")

def _split_into_sections(text: str) -> List[Tuple[str, List[str]]]:
    """
    Return list of (section_title, paragraphs[])
    Markdown-style headings win; fallback to blank-line paragraphs.
    """
    lines = text.splitlines()
    sections: List[Tuple[str, List[str]]] = []
    cur_title = "INTRO"
    cur_buf: List[str] = []

    def flush():
        nonlocal cur_title, cur_buf, sections
        if cur_buf:
            paras = [p.strip() for p in "\n".join(cur_buf).split("\n\n") if p.strip()]
            sections.append((cur_title, paras))
            cur_buf = []

    for ln in lines:
        m = _HEADING.match(ln.strip())
        if m:
            flush()
            cur_title = m.group(2).strip()
        else:
            cur_buf.append(ln)
    flush()
    if not sections:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        sections.append(("BODY", paras))
    return sections

def _chunk_paragraphs(paras: List[str], target_tokens: int = 400, overlap_tokens: int = 80) -> List[str]:
    """
    Very simple token-ish chunker by words length; good enough for now.
    """
    chunks = []
    buf, count = [], 0
    for p in paras:
        w = p.split()
        if count + len(w) > target_tokens and buf:
            chunks.append(" ".join(buf))
            # overlap
            back = " ".join(" ".join(buf).split()[-overlap_tokens:])
            buf, count = ([back] if back else []), len(back.split())
        buf.append(p)
        count += len(w)
    if buf:
        chunks.append(" ".join(buf))
    return chunks

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def build_payloads_from_file(file_path: Path, product: str = "domains", lang: str = "en") -> List[Dict[str, Any]]:
    text = _read_text_from_file(file_path)
    sections = _split_into_sections(text)
    doc_title = file_path.stem
    doc_id = f"{doc_title}-{uuid.uuid4().hex[:8]}"

    payloads: List[Dict[str, Any]] = []
    para_counter = 1

    for section_title, paras in sections:
        for chunk in _chunk_paragraphs(paras):
            anchor_id = f"para-{para_counter:04d}"
            payloads.append({
                "ref_id": f"{doc_id}-{anchor_id}",
                "doc": doc_title,
                "doc_id": doc_id,
                "section": section_title,
                "anchor_id": anchor_id,
                "text": chunk,
                "product": product,
                "lang": lang,
                "source_path": str(file_path),
                "updated_at": _now_iso(),
            })
            para_counter += 1
    return payloads

def ingest_file(file_path: Path, product: str = "domains", lang: str = "en", batch: int = 64) -> Dict[str, Any]:
    payloads = build_payloads_from_file(file_path, product=product, lang=lang)
    if not payloads:
        return {"ok": False, "reason": "no_chunks", "file": str(file_path)}

    ids = [str(uuid.uuid4()) for _ in payloads]

    for i, p in enumerate(payloads):
        p["id"] = ids[i]
        p["ref_id"] = f"{p['doc_id']}-{p['anchor_id']}"

    texts = [p["text"] for p in payloads]
    total = len(texts)
    for i in range(0, total, batch):
        vecs = embed_texts(texts[i:i+batch])
        qdrant_store.upsert(ids=ids[i:i+batch], vectors=vecs, payloads=payloads[i:i+batch])

    return {"ok": True, "file": str(file_path), "doc_id": payloads[0]["doc_id"], "chunks": len(payloads)}


def ingest_folder(folder: Path, product: str = "domains", lang: str = "en") -> List[Dict[str, Any]]:
    supported = (".md",".txt",".html",".htm",".pdf")
    results = []
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in supported:
            try:
                results.append(ingest_file(path, product=product, lang=lang))
            except Exception as e:
                results.append({"ok": False, "file": str(path), "error": str(e)})
    return results
