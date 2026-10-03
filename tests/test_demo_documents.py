"""Document upload changes the evidence used by the local UI, not pilot state."""

from __future__ import annotations

from io import BytesIO

import pytest

pytest.importorskip("pypdf")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from pypdf import PdfWriter  # noqa: E402
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject  # noqa: E402

from pramana.api.demo_documents import DocumentError, prepare_document  # noqa: E402
from pramana.api.service import app  # noqa: E402


def text_pdf(text: str) -> bytes:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
    })
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def client():
    from pramana.api import service
    from pramana.config.settings import Settings

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(service.Settings, "from_env", lambda: Settings(offline=True))
        with TestClient(app, base_url="http://localhost") as test_client:
            yield test_client


def test_pdf_extraction_preserves_page_source_and_hash():
    prepared = prepare_document(text_pdf("Appeals are allowed within 60 days."), "policy.pdf", "en")
    assert prepared.pages == 1
    assert prepared.chunks
    chunk = prepared.chunks[0]
    assert "60 days" in chunk.text
    assert chunk.metadata["page"] == 1
    assert chunk.metadata["source"] == "policy.pdf"
    assert chunk.metadata["sha256"] == prepared.sha256


def test_scanned_pdf_is_rejected():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buffer = BytesIO()
    writer.write(buffer)
    with pytest.raises(DocumentError, match="OCR"):
        prepare_document(buffer.getvalue(), "scan.pdf", "en")


def test_upload_audit_and_reset_are_consistent(client):
    original = client.get("/v1/corpus").json()["en"]
    uploaded = client.post(
        "/v1/demo/documents?filename=sample-policy.pdf&language=en",
        content=text_pdf("Rejected claims may be appealed within 60 days of the notice."),
        headers={"Content-Type": "application/pdf"},
    )
    assert uploaded.status_code == 200
    assert uploaded.json()["pages"] == 1
    corpus = client.get("/v1/corpus").json()["en"]
    assert all(c["source"] == "sample-policy.pdf" and c["page"] == 1 for c in corpus)
    assert {c["chunk_id"] for c in original}.isdisjoint({c["chunk_id"] for c in corpus})
    state = client.get("/v1/demo/documents").json()
    assert state["api_embedding_model"] == ""
    assert state["documents"]["en"]["name"] == "sample-policy.pdf"
    audit = client.post("/v1/verify", json={
        "query": "How long can I appeal a rejected claim?",
        "answer": "Rejected claims may be appealed within 90 days of the notice.",
        "language": "en",
    })
    assert audit.status_code == 200
    assert "CONTRADICTED" in {claim["verdict"] for claim in audit.json()["claims"]}
    assert all(cid in {c["chunk_id"] for c in corpus} for cid in audit.json()["retrieved_chunk_ids"])
    assert client.post("/v1/ask", json={"query": "When can I appeal?", "language": "en"}).status_code == 422
    assert client.delete("/v1/demo/documents").status_code == 200
    assert client.get("/v1/corpus").json()["en"] == original


def test_upload_validation_and_body_limit(client):
    assert client.post("/v1/demo/documents?filename=bad.pdf&language=en", content=b"not pdf").status_code == 422
    assert client.post("/v1/demo/documents?filename=bad.exe&language=en", content=b"x").status_code == 422
    assert client.post("/v1/demo/documents?filename=bad.txt&language=en", content=b"\xff").status_code == 422
    assert client.post("/v1/demo/documents?filename=big.txt&language=en", content=b"a" * 5_000_001).status_code == 413
    assert client.post("/v1/demo/documents?filename=policy.txt&language=en",
                       content=b"Only a demo policy.",
                       headers={"Origin": "https://unrelated.example"}).status_code == 403


def test_reset_selected_language_keeps_other_uploaded_document(client):
    assert client.post("/v1/demo/documents?filename=english.txt&language=en",
                       content=b"Appeals take 60 days.").status_code == 200
    assert client.post("/v1/demo/documents?filename=hindi.txt&language=hi",
                       content="अपील 60 दिनों के भीतर करें।".encode()).status_code == 200
    assert client.delete("/v1/demo/documents?language=en").status_code == 200
    documents = client.get("/v1/demo/documents").json()["documents"]
    assert documents["en"]["sample"]
    assert documents["hi"]["name"] == "hindi.txt"
