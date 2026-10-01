"""Bootstrap and safe local launch configuration; no installs or provider calls."""
from __future__ import annotations

import os
import socket
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest
import run_demo
import yaml
from scripts import serve_demo


@pytest.fixture
def clean_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("PRAMANA_") or key.endswith("API_KEY"):
            monkeypatch.delenv(key)


def plan(arguments):
    parser = serve_demo.make_parser()
    return serve_demo.resolve_plan(parser.parse_args(arguments), parser)


def test_google_automatically_enables_api_semantics(clean_environment, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-test-key")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-test-key")
    result = plan([])
    assert result.providers == ("google", "groq")
    assert result.semantic and result.port == 8765
    assert "fake-google-test-key" not in repr(result)


def test_groq_only_is_explicit_sparse_mode(clean_environment, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-test-key")
    result = plan([])
    assert result.providers == ("groq",) and not result.semantic


def test_no_key_never_silently_uses_fixtures(clean_environment, capsys):
    with pytest.raises(SystemExit) as error:
        plan([])
    assert error.value.code == 2
    assert "GOOGLE_API_KEY or GROQ_API_KEY" in capsys.readouterr().err


def test_offline_suppresses_google_embeddings(clean_environment, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-test-key")
    monkeypatch.setenv("PRAMANA_SEMANTIC", "true")
    result = plan(["--offline"])
    assert result.providers == ("stub",) and not result.semantic


@pytest.mark.parametrize("arguments", [
    ["--offline", "--semantic"], ["--offline", "--provider", "google"],
    ["--offline", "--live-google"], ["--semantic"],
])
def test_incompatible_or_unconfigured_options_fail_closed(clean_environment, arguments):
    with pytest.raises(SystemExit):
        plan(arguments)


def test_cli_overrides_configuration(clean_environment, monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-test-key")
    monkeypatch.setenv("PRAMANA_PORT", "9876")
    monkeypatch.setenv("PRAMANA_SEMANTIC", "true")
    result = plan(["--port", "8770", "--no-semantic"])
    assert result.port == 8770 and not result.semantic


@pytest.mark.parametrize("name,value", [("PRAMANA_PORT", "secret-invalid"),
                                        ("PRAMANA_PORT", "70000"),
                                        ("PRAMANA_SEMANTIC", "secret-invalid")])
def test_invalid_settings_do_not_print_values(clean_environment, monkeypatch, capsys, name, value):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-test-key")
    monkeypatch.setenv(name, value)
    with pytest.raises(SystemExit):
        plan([])
    assert "secret-invalid" not in capsys.readouterr().err


def test_launcher_clears_local_models_and_keeps_loopback_boundary(clean_environment, monkeypatch):
    monkeypatch.setenv("PRAMANA_DENSE_MODEL", "a-heavy-local-model")
    monkeypatch.setenv("PRAMANA_TRUSTED_HOSTS", "attacker.example")
    monkeypatch.setenv("PRAMANA_MODE", "pilot")
    monkeypatch.setenv("PRAMANA_API_KEY", "secret-test-token")
    serve_demo.configure_environment(serve_demo.LaunchPlan(8765, ("google",), True))
    assert os.environ["PRAMANA_DENSE_MODEL"] == ""
    assert os.environ["PRAMANA_TRUSTED_HOSTS"] == "127.0.0.1,localhost,::1"
    assert os.environ["PRAMANA_MODE"] == "demo"
    assert os.environ["PRAMANA_API_KEY"] == ""
    assert os.environ["PRAMANA_API_EMBEDDING_MODEL"] == "gemini-embedding-001"


def test_busy_port_is_detected():
    with socket.socket() as bound:
        bound.bind(("127.0.0.1", 0))
        bound.listen()
        assert not serve_demo.available_port(bound.getsockname()[1])


def test_configuration_created_only_when_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(run_demo, "ROOT", tmp_path)
    (tmp_path / ".env.example").write_text("GOOGLE_API_KEY=\n", encoding="utf-8")
    run_demo.ensure_configuration()
    config = tmp_path / ".env"
    assert config.read_text(encoding="utf-8") == "GOOGLE_API_KEY=\n"
    config.write_text("GOOGLE_API_KEY=existing-secret\n", encoding="utf-8")
    run_demo.ensure_configuration()
    assert config.read_text(encoding="utf-8") == "GOOGLE_API_KEY=existing-secret\n"


def test_doctor_does_not_install_or_create_configuration(monkeypatch, capsys):
    monkeypatch.setattr(run_demo, "dependencies_ready", lambda: False)
    install = Mock()
    configure = Mock()
    monkeypatch.setattr(run_demo, "install_dependencies", install)
    monkeypatch.setattr(run_demo, "ensure_configuration", configure)
    assert run_demo.main(["doctor"]) == 1
    install.assert_not_called()
    configure.assert_not_called()
    assert "setup" in capsys.readouterr().out


def test_setup_does_not_launch_or_spend_provider_quota(monkeypatch):
    monkeypatch.setattr(run_demo, "ensure_configuration", lambda: None)
    monkeypatch.setattr(run_demo, "install_dependencies", lambda: None)
    process = Mock()
    monkeypatch.setattr(run_demo.subprocess, "run", process)
    assert run_demo.main(["setup"]) == 0
    process.assert_not_called()


def test_run_and_doctor_forward_flags_without_keys(monkeypatch):
    monkeypatch.setattr(run_demo, "dependencies_ready", lambda: True)
    process = Mock(return_value=subprocess.CompletedProcess([], 0))
    monkeypatch.setattr(run_demo.subprocess, "run", process)
    assert run_demo.main(["doctor", "--offline", "--port", "8770"]) == 0
    command = process.call_args.args[0]
    assert "--check" in command and "--offline" in command and "8770" in command


def test_pip_failure_is_actionable_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(run_demo, "ensure_configuration", lambda: None)
    def fail():
        raise subprocess.CalledProcessError(1, ["pip", "secret-test-command"])
    monkeypatch.setattr(run_demo, "install_dependencies", fail)
    assert run_demo.main([]) == 1
    output = capsys.readouterr().out
    assert "rerun setup" in output and "secret-test-command" not in output


@pytest.mark.parametrize("arguments", [["--port", "0"], ["--offline", "--semantic"], ["--unknown-option"]])
def test_bootstrap_rejects_bad_arguments_before_install(monkeypatch, arguments):
    install = Mock()
    monkeypatch.setattr(run_demo, "install_dependencies", install)
    with pytest.raises(SystemExit):
        run_demo.main(arguments)
    install.assert_not_called()


def test_existing_environment_is_not_reinstalled(monkeypatch, tmp_path):
    executable = tmp_path / "python"
    executable.touch()
    monkeypatch.setattr(run_demo, "PYTHON", executable)
    monkeypatch.setattr(run_demo, "dependencies_ready", lambda: True)
    process = Mock()
    monkeypatch.setattr(run_demo.subprocess, "run", process)
    run_demo.install_dependencies()
    process.assert_not_called()


def test_native_host_always_remains_loopback():
    assert serve_demo.bind_host(False) == "127.0.0.1"


def test_container_bind_rejected_outside_docker(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: False)
    with pytest.raises(ValueError, match="only allowed inside Docker"):
        serve_demo.bind_host(True)


def test_container_bind_allowed_only_with_marker(monkeypatch):
    monkeypatch.setattr(Path, "is_file", lambda self: self.as_posix() == "/.dockerenv")
    assert serve_demo.bind_host(True) == "0.0.0.0"


def test_live_docker_profile_is_hardened_and_self_contained():
    root = Path(__file__).resolve().parents[1]
    service = yaml.safe_load((root / "compose.local.yaml").read_text())["services"]["pramana"]
    assert service["read_only"] is True
    assert service["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in service["security_opt"]
    assert service["restart"] == "unless-stopped"
    assert service["pull_policy"] == "never"
    assert service["ports"][0].startswith("127.0.0.1:")
    assert "--container" in service["command"]
    assert service["volumes"] == ["local-state:/app/data"]
    assert "COPY examples/corpus ./examples/corpus" in (root / "Dockerfile").read_text()


def test_live_docker_does_not_forward_an_entire_secret_file():
    root = Path(__file__).resolve().parents[1]
    service = yaml.safe_load((root / "compose.local.yaml").read_text())["services"]["pramana"]
    assert "env_file" not in service
    assert {k for k in service["environment"] if k.endswith("API_KEY")} == {"GOOGLE_API_KEY", "GROQ_API_KEY"}


def test_other_docker_modes_remain_explicit_and_separate():
    root = Path(__file__).resolve().parents[1]
    pilot = yaml.safe_load((root / "compose.yaml").read_text())["services"]["pramana"]
    offline = yaml.safe_load((root / "compose.demo.yaml").read_text())["services"]["pramana"]
    assert pilot["environment"]["PRAMANA_MODE"] == "pilot"
    assert offline["environment"]["PRAMANA_OFFLINE"] == "true"
