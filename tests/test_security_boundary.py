"""Local-host protections are independent of LLM factual quality."""
import pytest
from fastapi.testclient import TestClient

from pramana.api import service
from pramana.api.security import trusted_authority
from pramana.config.settings import Settings


@pytest.mark.parametrize("host", [b"127.0.0.1:8000", b"localhost", b"LOCALHOST:8765", b"[::1]:8000"])
def test_local_authorities_are_accepted(host):
    assert trusted_authority(host, Settings().trusted_hosts)


@pytest.mark.parametrize("host", [b"attacker.example", b"localhost.attacker.example", b"user@localhost",
                                   b"localhost:bad", b"localhost:99999", b"localhost/path", b"localhost#x",
                                   b"localhost?x", b"localhost\n", b"", b"\xff", b"[::1"])
def test_rebinding_and_malformed_authorities_are_rejected(host):
    assert not trusted_authority(host, Settings().trusted_hosts)


@pytest.mark.parametrize("hosts", [(), ("*",), ("localhost:8000",), ("https://example.com",)])
def test_host_configuration_requires_exact_explicit_hosts(hosts):
    with pytest.raises(ValueError, match="TRUSTED_HOSTS"):
        Settings(trusted_hosts=hosts)


def test_matching_attacker_origin_cannot_bypass_demo_boundary(monkeypatch):
    monkeypatch.setattr(service.Settings, "from_env", lambda: Settings(offline=True))
    with TestClient(service.app, base_url="http://localhost") as client:
        response = client.post("/v1/ask", json={"query": "appeals"},
                               headers={"Host": "attacker.example", "Origin": "http://attacker.example"})
        assert response.status_code == 400
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["referrer-policy"] == "no-referrer"
