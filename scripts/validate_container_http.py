"""Offline HTTP checks for an isolated loopback demo container; no live API calls.

Uses only fictional sample documents. Refuses online/pilot services and records
actual responses, not human accuracy or production acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from validate_live import CASES


def fictional_pdf() -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                             NameObject("/Subtype"): NameObject("/Type1"),
                             NameObject("/BaseFont"): NameObject("/Helvetica")})
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
        NameObject("/F1"): writer._add_object(font)})})
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (Rejected claims may be appealed within 60 days of the notice.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    address = urlsplit(args.url)
    if (address.scheme != "http" or address.hostname not in {"127.0.0.1", "localhost", "::1"}
            or address.username or address.password or address.path not in {"", "/"}
            or address.query or address.fragment):
        parser.error("Use an explicit loopback HTTP origin")
    root = Path(__file__).resolve().parents[1]
    target = args.out.resolve()
    if not target.is_relative_to(root) or target.exists():
        parser.error("Use a fresh report path inside the project")
    report = {"timestamp": datetime.now(UTC).isoformat(), "scope": "offline_container_http",
              "live_api_tested": False, "human_accuracy_tested": False,
              "url": args.url, "checks": [], "complete": False}
    target.parent.mkdir(parents=True, exist_ok=True)

    def check(name: str, passed: bool, response: httpx.Response | None = None):
        row = {"name": name, "passed": bool(passed)}
        if response is not None:
            row["status"] = response.status_code
            if "application/json" in response.headers.get("content-type", ""):
                row["response"] = response.json()
        report["checks"].append(row)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{name}: {'PASS' if passed else 'FAIL'}", flush=True)

    with httpx.Client(base_url=args.url, timeout=30, trust_env=False) as client:
        ready = client.get("/v1/ready")
        state = ready.json()
        if ready.status_code != 200 or not state.get("offline") or state.get("mode") != "demo":
            parser.error("Refusing to probe a non-offline demo; no inference/upload was attempted")
        check("ready_offline_demo", True, ready)
        health = client.get("/v1/health")
        check("health", health.status_code == 200 and health.json()["status"] == "ok", health)
        check("security_headers", all(health.headers.get(k) == v for k, v in {
            "cache-control": "no-store", "x-content-type-options": "nosniff",
            "x-frame-options": "DENY", "referrer-policy": "no-referrer"}.items()))
        page = client.get("/")
        check("demo_page", page.status_code == 200 and "PRAMANA" in page.text)
        schema = client.get("/openapi.json")
        check("openapi", schema.status_code == 200 and "/v1/verify" in schema.json()["paths"])
        hostile = client.get("/v1/health", headers={"Host": "attacker.example"})
        check("untrusted_host_blocked", hostile.status_code == 400, hostile)
        hostile = client.post("/v1/ask", json={"query": "claims"}, headers={"Origin": "https://attacker.example"})
        check("cross_origin_blocked", hostile.status_code == 403, hostile)
        invalid = client.post("/v1/ask", json={"query": ""})
        check("empty_query_rejected", invalid.status_code == 422, invalid)
        invalid = client.post("/v1/ask", content=b"not json", headers={"Content-Type": "application/json"})
        check("malformed_json_rejected", invalid.status_code == 422, invalid)
        oversized = client.post("/v1/verify", content=b"x" * 32769)
        check("body_limit", oversized.status_code == 413, oversized)
        corpus_response = client.get("/v1/corpus")
        corpus = corpus_response.json()
        check("three_language_corpus", corpus_response.status_code == 200 and set(corpus) == set(CASES))
        for lang, case in CASES.items():
            known = {c["chunk_id"] for c in corpus[lang]}
            response = client.post("/v1/ask", json={"query": case["query"], "language": lang})
            body = response.json()
            citations = body.get("evidence_chunk_ids", [])
            # Stub drafts are deliberately planted-error fixtures, not correct
            # multilingual answers. Check the HTTP contract, not live accuracy.
            claims = body.get("claims", [])
            passed = (response.status_code == 200 and body.get("detected_language") == lang
                      and isinstance(body.get("answer"), str) and bool(body.get("trace_id"))
                      and 0 <= body.get("confidence", {}).get("score", -1) <= 1
                      and set(citations).issubset(known)
                      and all(c["verdict"] in {"SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"}
                              and set(c["evidence"]).issubset(known) for c in claims)
                      and (not body.get("abstained") or not claims))
            check(f"{lang}_offline_answer_contract", passed, response)
            for name, verdict in (("supported", "SUPPORTED"), ("contradiction", "CONTRADICTED")):
                response = client.post("/v1/verify", json={"query": case["query"], "answer": case[name], "language": lang})
                passed = response.status_code == 200 and verdict in {c["verdict"] for c in response.json().get("claims", [])}
                check(f"{lang}_{name}", passed, response)
            response = client.post("/v1/ask", json={"query": case["missing"], "language": lang})
            check(f"{lang}_missing_abstains", response.status_code == 200 and response.json().get("abstained"), response)
        document = fictional_pdf()
        uploaded = False
        try:
            response = client.post("/v1/demo/documents", params={"filename": "docker-fictional-policy.pdf", "language": "en"},
                                   content=document, headers={"Content-Type": "application/pdf"})
            uploaded = response.status_code == 200
            check("pdf_upload", uploaded and response.json().get("pages") == 1, response)
            if uploaded:
                chunks = client.get("/v1/corpus").json()["en"]
                check("pdf_provenance", bool(chunks) and all(c["source"] == "docker-fictional-policy.pdf"
                      and c["page"] == 1 and c["sha256"] == hashlib.sha256(document).hexdigest() for c in chunks))
                response = client.post("/v1/verify", json={"query": CASES["en"]["query"],
                                       "answer": CASES["en"]["contradiction"], "language": "en"})
                check("uploaded_pdf_contradiction", response.status_code == 200 and "CONTRADICTED" in
                      {c["verdict"] for c in response.json().get("claims", [])}, response)
                response = client.post("/v1/ask", json={"query": CASES["en"]["query"], "language": "en"})
                check("offline_uploaded_generation_refused", response.status_code == 422, response)
        finally:
            if uploaded:
                response = client.delete("/v1/demo/documents", params={"language": "en"})
                check("pdf_reset", response.status_code == 200 and client.get("/v1/corpus").json()["en"] == corpus["en"], response)
        response = client.get("/v1/runtime")
        runtime = response.json()
        check("only_stub_no_embedding_api", response.status_code == 200 and runtime.get("offline")
              and runtime.get("embedding_api_requests") == 0
              and [p["name"] for p in runtime.get("providers", [])] == ["stub"], response)
    report["complete"] = True
    report["passed"] = all(c["passed"] for c in report["checks"])
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
