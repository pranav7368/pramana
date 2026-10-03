"""Hosted public demo: per-visitor uploads, usage limits and the platform boundary.

All offline; no key and no network. These protect the properties that make a
public URL safe to share: one visitor cannot see or replace another visitor's
document, and anonymous traffic cannot spend more than the configured budget.
"""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest

pytest.importorskip("fastapi", reason="API extra not installed")
from fastapi.testclient import TestClient
from scripts import serve_demo

from pramana.api import service
from pramana.api.quota import UsageLimiter
from pramana.api.sessions import SessionStore, session_id
from pramana.config.settings import Settings

HOST = "pramana-demo.example.com"
SESSION_A = {"X-Pramana-Session": "a" * 32}
SESSION_B = {"X-Pramana-Session": "b" * 32}
DOC = "# Travel policy\n\nEmployees may claim a daily meal allowance of 900 rupees on approved trips.\n"


def public_settings(**overrides) -> Settings:
    values = dict(mode="public", offline=True, trusted_hosts=("localhost", HOST),
                  client_ip_header="true-client-ip", visitor_limit=50, daily_limit=500)
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def make_client(monkeypatch):
    def start(**overrides):
        settings = public_settings(**overrides)
        monkeypatch.setattr(service.Settings, "from_env", lambda: settings)
        return TestClient(service.app, base_url=f"https://{HOST}")
    return start


def upload(client, headers, name="travel.md", body=DOC, language="en"):
    return client.post(f"/v1/demo/documents?filename={name}&language={language}",
                       content=body.encode(), headers={**headers, "Content-Type": "text/markdown"})


class TestSettings:
    @pytest.fixture
    def clean(self, monkeypatch):
        for key in tuple(os.environ):
            if key.startswith(("PRAMANA_", "RENDER", "SPACE_")):
                monkeypatch.delenv(key)

    def test_render_hostname_and_client_header_are_derived(self, clean, monkeypatch):
        monkeypatch.setenv("PRAMANA_MODE", "public")
        monkeypatch.setenv("RENDER", "true")
        monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "pramana.onrender.com")
        cfg = Settings.from_env()
        assert "pramana.onrender.com" in cfg.trusted_hosts and "127.0.0.1" in cfg.trusted_hosts
        assert cfg.client_ip_header == "true-client-ip"
        assert cfg.visitor_limit > 0 and cfg.daily_limit > 0 and cfg.queue_wait_s > 0
        assert cfg.cache_enabled is False

    def test_platform_hosts_are_ignored_outside_public_mode(self, clean, monkeypatch):
        monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "pramana.onrender.com")
        cfg = Settings.from_env()
        assert cfg.trusted_hosts == ("127.0.0.1", "localhost", "::1")
        assert cfg.visitor_limit == 0 and cfg.daily_limit == 0

    def test_public_mode_refuses_unsafe_combinations(self):
        with pytest.raises(ValueError, match="PRAMANA_API_KEY"):
            Settings(mode="public", api_key="x" * 40)
        with pytest.raises(ValueError, match="cache"):
            Settings(mode="public", cache_enabled=True)
        with pytest.raises(ValueError, match="daily_limit"):
            Settings(mode="public", daily_limit=-1)
        with pytest.raises(ValueError, match="CLIENT_IP_HEADER"):
            Settings(mode="public", client_ip_header="X Forwarded")


class TestUsageLimiter:
    def test_visitor_window_blocks_then_recovers(self):
        now = [0.0]
        limiter = UsageLimiter(visitor_limit=2, window_s=60, clock=lambda: now[0])
        assert limiter.consume("a").allowed and limiter.consume("a").allowed
        blocked = limiter.consume("a")
        assert not blocked.allowed and blocked.retry_after_s > 0 and "2 checks" in blocked.detail
        assert limiter.consume("b").allowed
        now[0] = 61.0
        assert limiter.consume("a").allowed

    def test_daily_budget_is_shared_and_resets_at_utc_midnight(self):
        day = [datetime(2026, 10, 4, 23, 0, tzinfo=UTC)]
        limiter = UsageLimiter(daily_limit=2, wall=lambda: day[0])
        assert limiter.consume("a").allowed and limiter.consume("b").allowed
        blocked = limiter.consume("c")
        assert not blocked.allowed and blocked.retry_after_s == 3600 and "00:00 UTC" in blocked.detail
        assert limiter.remaining("c")["daily_remaining"] == 0
        day[0] += timedelta(hours=2)
        assert limiter.consume("c").allowed

    def test_disabled_limiter_reports_nothing(self):
        assert not UsageLimiter().enabled


class TestSessionStore:
    def test_sessions_expire_and_evict_least_recent(self):
        now = [0.0]
        store = SessionStore(max_sessions=2, ttl_s=100, clock=lambda: now[0])
        for key in ("one", "two", "three"):
            store.put(key, "en", object(), object())
        assert store.get("one") is None and store.get("three") is not None
        now[0] = 500.0
        assert store.get("three") is None and len(store) == 0

    def test_a_new_upload_replaces_the_active_document(self):
        store = SessionStore()
        store.put("k", "en", "en-index", "en-doc")
        store.put("k", "hi", "hi-index", "hi-doc")
        found = store.get("k")
        assert found is not None and found.language == "hi" and set(found.documents) == {"hi"}
        store.reset("k", "en")
        assert store.get("k") is not None
        store.reset("k")
        assert store.get("k") is None

    @pytest.mark.parametrize("raw", [None, "", "short", "x" * 65, "has space" * 3, "../../etc/passwd/xxxxx"])
    def test_malformed_session_ids_are_refused(self, raw):
        assert session_id(raw) is None


class TestPublicService:
    def test_uploads_are_private_to_the_uploading_session(self, make_client):
        with make_client() as client:
            assert upload(client, SESSION_A).status_code == 200
            mine = client.get("/v1/demo/documents", headers=SESSION_A).json()["documents"]["en"]
            theirs = client.get("/v1/demo/documents", headers=SESSION_B).json()["documents"]["en"]
            assert mine["name"] == "travel.md" and theirs.get("sample") is True
            corpus_a = {c["text"] for c in client.get("/v1/corpus", headers=SESSION_A).json()["en"]}
            corpus_b = {c["text"] for c in client.get("/v1/corpus", headers=SESSION_B).json()["en"]}
            assert any("meal allowance" in t for t in corpus_a)
            assert not any("meal allowance" in t for t in corpus_b)
            # Offline fixtures cannot answer an uploaded document, but the other visitor is unaffected.
            assert client.post("/v1/ask", json={"query": "meal allowance", "language": "en"},
                               headers=SESSION_A).status_code == 422
            assert client.post("/v1/ask", json={"query": "When are claims rejected after discharge?"},
                               headers=SESSION_B).status_code == 200
            assert client.delete("/v1/demo/documents?language=en", headers=SESSION_A).status_code == 200
            restored = client.get("/v1/demo/documents", headers=SESSION_A).json()["documents"]["en"]
            assert restored.get("sample") is True

    def test_public_upload_requires_a_session(self, make_client):
        with make_client() as client:
            assert upload(client, {}).status_code == 400
            assert upload(client, {"X-Pramana-Session": "bad id"}).status_code == 400

    def test_https_same_origin_is_allowed_and_cross_origin_refused(self, make_client):
        with make_client() as client:
            ok = client.post("/v1/verify", json={"query": "appeal", "answer": "Appeals take 30 days."},
                             headers={"Origin": f"https://{HOST}"})
            assert ok.status_code == 200
            assert client.post("/v1/verify", json={"query": "appeal", "answer": "x"},
                               headers={"Origin": "https://attacker.example"}).status_code == 403

    def test_visitor_limit_returns_429_per_client_address(self, make_client):
        with make_client(visitor_limit=2) as client:
            first = {"true-client-ip": "203.0.113.1"}
            for _ in range(2):
                assert client.post("/v1/verify", json={"query": "appeal", "answer": "30 days."},
                                   headers=first).status_code == 200
            blocked = client.post("/v1/verify", json={"query": "appeal", "answer": "30 days."}, headers=first)
            assert blocked.status_code == 429 and int(blocked.headers["retry-after"]) > 0
            assert client.post("/v1/verify", json={"query": "appeal", "answer": "30 days."},
                               headers={"true-client-ip": "203.0.113.2"}).status_code == 200
            limits = client.get("/v1/demo/documents", headers=first).json()["limits"]
            assert limits["visitor_remaining"] == 0 and limits["visitor_limit"] == 2

    def test_daily_budget_stops_every_visitor(self, make_client):
        with make_client(daily_limit=1) as client:
            assert client.post("/v1/verify", json={"query": "appeal", "answer": "30 days."},
                               headers={"true-client-ip": "203.0.113.1"}).status_code == 200
            blocked = client.post("/v1/ask", json={"query": "appeal"}, headers={"true-client-ip": "203.0.113.9"})
            assert blocked.status_code == 429 and "today" in blocked.json()["detail"]
            assert client.get("/v1/runtime").json()["usage"]["used_today"] == 1

    def test_platform_health_probe_may_use_an_internal_host(self, make_client):
        with make_client() as client:
            assert client.get("/v1/health", headers={"Host": "10.0.0.7:8000"}).status_code == 200
            assert client.get("/v1/corpus", headers={"Host": "10.0.0.7:8000"}).status_code == 400
            assert client.get("/", headers={"Host": "attacker.example"}).status_code == 400

    def test_demo_page_is_served_and_reports_public_mode(self, make_client):
        with make_client() as client:
            page = client.get("/")
            assert page.status_code == 200 and "X-Pramana-Session" in page.text
            info = client.get("/v1/demo/documents").json()
            assert info["public"] is True and info["limits"]["daily_limit"] == 500


class TestPublicLauncher:
    @pytest.fixture
    def clean(self, monkeypatch):
        for key in tuple(os.environ):
            if key.startswith("PRAMANA_") or key.endswith("API_KEY") or key == "PORT":
                monkeypatch.delenv(key)

    def plan(self, arguments):
        parser = serve_demo.make_parser()
        return serve_demo.resolve_plan(parser.parse_args(arguments), parser)

    def test_public_reads_platform_port_and_falls_back_to_labelled_fixtures(self, clean, monkeypatch, capsys):
        monkeypatch.setenv("PORT", "10000")
        result = self.plan(["--public"])
        assert result.public and result.port == 10000
        assert result.providers == ("stub",) and not result.semantic
        assert "offline fixtures" in capsys.readouterr().out

    def test_public_uses_live_keys_when_present(self, clean, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-test-key")
        result = self.plan(["--public"])
        assert result.providers == ("google",) and result.semantic and result.port == 8000

    def test_local_launch_without_keys_still_refuses(self, clean):
        with pytest.raises(SystemExit):
            self.plan([])

    def test_public_binds_all_interfaces_without_docker_marker(self):
        assert serve_demo.bind_host(False, public=True) == "0.0.0.0"

    def test_public_environment_keeps_explicit_hosts_and_never_a_bearer_token(self, clean, monkeypatch):
        monkeypatch.setenv("PRAMANA_TRUSTED_HOSTS", HOST)
        monkeypatch.setenv("PRAMANA_API_KEY", "should-be-cleared")
        serve_demo.configure_environment(serve_demo.LaunchPlan(8000, ("stub",), False, public=True))
        assert os.environ["PRAMANA_MODE"] == "public"
        assert os.environ["PRAMANA_TRUSTED_HOSTS"] == HOST
        assert os.environ["PRAMANA_API_KEY"] == ""
        assert os.environ["PRAMANA_CACHE_ENABLED"] == "false"

    def test_public_environment_derives_hosts_from_the_platform(self, clean, monkeypatch):
        serve_demo.configure_environment(serve_demo.LaunchPlan(8000, ("stub",), False, public=True))
        assert "PRAMANA_TRUSTED_HOSTS" not in os.environ
        monkeypatch.setenv("RENDER_EXTERNAL_HOSTNAME", "pramana.onrender.com")
        assert "pramana.onrender.com" in Settings.from_env().trusted_hosts
