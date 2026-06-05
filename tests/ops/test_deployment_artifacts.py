"""Security + correctness lockers for the deployment artefacts.

``deploy/Dockerfile.api``, ``deploy/Dockerfile.frontend``,
``deploy/docker-entrypoint.sh``, the repo-root ``.dockerignore`` and
``frontend/.dockerignore`` are operator artefacts, not code — so nothing stops
them silently regressing into an insecure image. These static tests are the
load-bearing control for both deployment images:

    - **Supply chain.** Every base image MUST be digest-pinned
      (``@sha256:<64 hex>``) — mirrors the SHA-pinned GitHub Actions in
      ``ci.yml``. A floating ``:tag`` is a silent-substitution vector.
    - **Least privilege.** The long-lived server MUST run non-root. The image
      uses the gosu pattern: start as root only to chown the volume, then
      ``exec gosu`` to an unprivileged account. Both halves are asserted.
    - **No baked secret.** Neither the Dockerfile nor the entrypoint may carry
      secret-shaped material, and the build MUST NOT slurp the whole context
      (which would drag ``.env`` into a layer). ``.dockerignore`` MUST exclude
      every secret-/state-bearing path.
        - **Operability.** A ``HEALTHCHECK`` probes the unauth ``/api/health``; the
            server runs exactly one worker with ``--proxy-headers`` (the in-process
            TaskManager single-replica contract).

All assertions are on tracked, CI-visible files; nothing here requires Docker
or a network, so the lockers run in the normal pytest job.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE_API = _REPO_ROOT / "deploy" / "Dockerfile.api"
_DOCKERFILE_FRONTEND = _REPO_ROOT / "deploy" / "Dockerfile.frontend"
_ENTRYPOINT = _REPO_ROOT / "deploy" / "docker-entrypoint.sh"
_DOCKERIGNORE = _REPO_ROOT / ".dockerignore"
_FRONTEND_DOCKERIGNORE = _REPO_ROOT / "frontend" / ".dockerignore"
_NEXT_CONFIG = _REPO_ROOT / "frontend" / "next.config.ts"

_DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}\b")
_ARG_RE = re.compile(r"^ARG\s+([A-Za-z_][A-Za-z0-9_]*)=(.+)$")
_FROM_RE = re.compile(r"^FROM\s+(\S+)")
_VAR_RE = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")


def _collect_arg_defaults(dockerfile: str) -> dict[str, str]:
    """Map ``ARG NAME=default`` declarations to their default value."""
    defaults: dict[str, str] = {}
    for line in dockerfile.splitlines():
        match = _ARG_RE.match(line.strip())
        if match:
            defaults[match.group(1)] = match.group(2).strip()
    return defaults


def _collect_base_images(dockerfile: str) -> list[str]:
    """Every base reference on a ``FROM`` line, resolved through ARG defaults."""
    arg_defaults = _collect_arg_defaults(dockerfile)
    resolved: list[str] = []
    for line in dockerfile.splitlines():
        match = _FROM_RE.match(line.strip())
        if not match:
            continue
        token = match.group(1)
        var = _VAR_RE.match(token)
        if var:
            assert var.group(1) in arg_defaults, (
                f"FROM references ${var.group(1)} but no ARG default defines it"
            )
            token = arg_defaults[var.group(1)]
        resolved.append(token)
    return resolved


@pytest.fixture(scope="module")
def dockerfile() -> str:
    return _DOCKERFILE_API.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def entrypoint() -> str:
    return _ENTRYPOINT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def dockerignore() -> str:
    return _DOCKERIGNORE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontend_dockerfile() -> str:
    return _DOCKERFILE_FRONTEND.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def frontend_dockerignore() -> str:
    return _FRONTEND_DOCKERIGNORE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def next_config() -> str:
    return _NEXT_CONFIG.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def arg_defaults(dockerfile: str) -> dict[str, str]:
    return _collect_arg_defaults(dockerfile)


@pytest.fixture(scope="module")
def base_images(dockerfile: str) -> list[str]:
    return _collect_base_images(dockerfile)


# ---------------------------------------------------------------------------
# Files exist
# ---------------------------------------------------------------------------


def test_deployment_artifacts_exist() -> None:
    assert _DOCKERFILE_API.is_file(), "deploy/Dockerfile.api is missing"
    assert _ENTRYPOINT.is_file(), "deploy/docker-entrypoint.sh is missing"
    assert _DOCKERIGNORE.is_file(), ".dockerignore is missing"


# ---------------------------------------------------------------------------
# Supply chain: digest-pinned bases
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_every_base_image_is_digest_pinned(base_images: list[str]) -> None:
    assert base_images, "no FROM instruction found in deploy/Dockerfile.api"
    floating = [image for image in base_images if not _DIGEST_RE.search(image)]
    assert not floating, (
        f"base image(s) not digest-pinned (@sha256:...): {floating}. "
        "A floating :tag is a silent-substitution / supply-chain vector."
    )


# ---------------------------------------------------------------------------
# Least privilege: non-root server via the gosu drop pattern
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_image_creates_unprivileged_user(dockerfile: str) -> None:
    assert re.search(r"^RUN\s+useradd\b", dockerfile, re.MULTILINE) or re.search(
        r"^RUN\s+adduser\b", dockerfile, re.MULTILINE
    ), "Dockerfile does not create an unprivileged service account"


@pytest.mark.security
def test_entrypoint_drops_privileges_via_gosu(entrypoint: str) -> None:
    # The final, long-lived process must not be root: the entrypoint execs
    # the server through gosu after the privileged chown.
    assert "gosu" in entrypoint, "entrypoint never drops privileges via gosu"
    assert re.search(r"\bexec\s+gosu\b", entrypoint), (
        "entrypoint must 'exec gosu <user> \"$@\"' so the non-root server is PID 1"
    )
    # And the fallback path (already non-root) must still exec, not fork.
    assert re.search(r"\bexec\s+\"\$@\"", entrypoint), (
        "entrypoint must exec the command so signals reach the server directly"
    )


@pytest.mark.security
def test_dockerfile_does_not_pin_a_root_runtime_user(dockerfile: str) -> None:
    # We intentionally do NOT set `USER appuser` (gosu drops at runtime), but a
    # stray `USER root` as the last USER directive would defeat the pattern.
    user_lines = re.findall(r"^USER\s+(\S+)", dockerfile, re.MULTILINE)
    assert "root" not in user_lines, "Dockerfile pins USER root — defeats the gosu drop"


# ---------------------------------------------------------------------------
# No baked secret
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_no_baked_secret_in_image_definition(dockerfile: str, entrypoint: str) -> None:
    blob = f"{dockerfile}\n{entrypoint}"
    lowered = blob.lower()
    # Obvious secret-shaped tokens. The sk- check is anchored on a word
    # boundary + key-length tail so it flags `sk-proj-...` keys, not the
    # `sk-` buried inside an English word (e.g. "ta[sk-]404s").
    assert not re.search(r"\bsk-[a-z0-9]{8,}", lowered), (
        "Dockerfile/entrypoint contains an sk- API-key-shaped token"
    )
    for needle in ("bearer ", "authorization:", "private key"):
        assert needle not in lowered, (
            f"image definition contains secret-shaped material: {needle!r}"
        )
    # Secret-bearing env knobs must never be ASSIGNED a literal value in the
    # image (referencing them at runtime is fine; only `NAME=<value>` is the
    # leak). Allow `NAME=` empty and `NAME=${VAR}` indirections.
    secret_env = (
        "API_KEY",
        "API_ADMIN_KEY",
        "API_AUTH_PEPPER",
        "DB_ENCRYPTION_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "HUGGING_FACE_TOKEN",
    )
    for name in secret_env:
        bad = re.search(rf"\b{name}=(?!\s|$|\$)\S", blob)
        assert not bad, f"image definition assigns a literal value to {name} — never bake a secret"


@pytest.mark.security
def test_dockerfile_does_not_copy_whole_context(dockerfile: str) -> None:
    # `COPY . .` / `ADD . .` would drag the entire context (incl. a stray
    # `.env`) into a layer. Selective COPYs only.
    for line in dockerfile.splitlines():
        stripped = line.strip()
        assert not re.match(r"^(COPY|ADD)\s+\.(\s|/|$)", stripped), (
            f"image copies the whole build context: {stripped!r}. Copy explicit paths only."
        )


# ---------------------------------------------------------------------------
# .dockerignore excludes every secret-/state-bearing path
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_dockerignore_excludes_secrets_and_state(dockerignore: str) -> None:
    entries = {line.strip() for line in dockerignore.splitlines() if line.strip()}
    required = {
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "edgar-identity.txt",
        "data/",
        ".git/",
        ".venv/",
    }
    missing = sorted(required - entries)
    assert not missing, f".dockerignore is missing secret/state exclusions: {missing}"


# ---------------------------------------------------------------------------
# Operability: healthcheck + single-worker proxy-aware server
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_healthcheck_probes_health_endpoint(dockerfile: str) -> None:
    assert "HEALTHCHECK" in dockerfile, "Dockerfile has no HEALTHCHECK"
    assert "/api/health" in dockerfile, "HEALTHCHECK does not probe /api/health"


def test_server_runs_single_worker_behind_proxy(dockerfile: str) -> None:
    # Collapse the CMD continuation lines so flag/value pairs are adjacent.
    flat = re.sub(r"\\\s*\n", " ", dockerfile)
    cmd = next((ln for ln in flat.splitlines() if ln.strip().startswith("CMD")), "")
    assert "uvicorn" in cmd, "CMD does not launch uvicorn"
    assert "create_app" in cmd and "--factory" in cmd, "CMD does not use the create_app factory"
    assert re.search(r'"--workers",\s*"1"', cmd), (
        "uvicorn must run exactly one worker (in-process TaskManager contract)"
    )
    assert "--proxy-headers" in cmd, "uvicorn must run with --proxy-headers behind nginx/GFE"


# ==========================================================================
# Frontend image (deploy/Dockerfile.frontend) — the frontend-image-keyless
# half of the deployment lockers.
# ===========================================================================


def test_frontend_artifacts_exist() -> None:
    assert _DOCKERFILE_FRONTEND.is_file(), "deploy/Dockerfile.frontend is missing"
    assert _FRONTEND_DOCKERIGNORE.is_file(), "frontend/.dockerignore is missing"


@pytest.mark.security
def test_frontend_every_base_image_is_digest_pinned(frontend_dockerfile: str) -> None:
    base_images = _collect_base_images(frontend_dockerfile)
    assert base_images, "no FROM instruction found in deploy/Dockerfile.frontend"
    floating = [image for image in base_images if not _DIGEST_RE.search(image)]
    assert not floating, (
        f"frontend base image(s) not digest-pinned (@sha256:...): {floating}. "
        "A floating :tag is a silent-substitution / supply-chain vector."
    )


@pytest.mark.security
def test_frontend_runs_non_root(frontend_dockerfile: str) -> None:
    # The frontend is stateless (no mounted data volume), so there is no gosu
    # chown dance — the image must instead drop to a non-root user with a plain
    # `USER` directive, and the LAST such directive must not be root.
    user_lines = re.findall(r"^USER\s+(\S+)", frontend_dockerfile, re.MULTILINE)
    assert user_lines, "frontend Dockerfile never sets a non-root USER"
    assert user_lines[-1] != "root", "frontend Dockerfile's final USER is root"
    assert user_lines[-1] not in {"0", "0:0"}, "frontend Dockerfile's final USER is uid 0"


@pytest.mark.security
def test_frontend_image_is_keyless(frontend_dockerfile: str) -> None:
    # The frontend bakes NO secret. The operator/admin keys live only in the
    # Next server-side admin-session Map (minted at runtime via WelcomeGate);
    # the only runtime env knob is SEC_API_BASE_URL, and even that is supplied
    # at run time, never assigned a literal in the image.
    lowered = frontend_dockerfile.lower()
    assert not re.search(r"\bsk-[a-z0-9]{8,}", lowered), (
        "frontend Dockerfile contains an sk- API-key-shaped token"
    )
    for needle in ("bearer ", "authorization:", "private key"):
        assert needle not in lowered, (
            f"frontend image definition contains secret-shaped material: {needle!r}"
        )
    # No secret-bearing env knob may be ASSIGNED a literal value. A bare
    # `NAME=` or a `NAME=${VAR}` indirection is fine; `NAME=<value>` is a leak.
    # NEXT_PUBLIC_* is doubly forbidden a key — it would ship to the browser.
    secret_env = (
        "API_KEY",
        "API_ADMIN_KEY",
        "API_AUTH_PEPPER",
        "DB_ENCRYPTION_KEY",
        "SEC_API_BASE_URL",
        "NEXT_PUBLIC_API_KEY",
        "NEXT_PUBLIC_ADMIN_KEY",
        "NEXT_PUBLIC_SEC_API_BASE_URL",
    )
    for name in secret_env:
        bad = re.search(rf"\b{name}=(?!\s|$|\$)\S", frontend_dockerfile)
        assert not bad, f"frontend image assigns a literal value to {name} — never bake a key / URL"
    # Belt-and-braces: the admin key must NEVER be shipped to the browser via a
    # NEXT_PUBLIC_ env var of any name.
    assert not re.search(r"\bNEXT_PUBLIC_\w*(?:ADMIN|API_KEY|PEPPER)", frontend_dockerfile), (
        "frontend image exposes a key via a NEXT_PUBLIC_* env var — reaches the browser"
    )


@pytest.mark.security
def test_frontend_does_not_copy_whole_context(frontend_dockerfile: str) -> None:
    # `COPY . .` / `ADD . .` would drag the entire context (incl. a stray
    # `.env`) into a build layer. Selective COPYs only; `.dockerignore` is a
    # backstop, not the only line of defence.
    for line in frontend_dockerfile.splitlines():
        stripped = line.strip()
        assert not re.match(r"^(COPY|ADD)\s+\.(\s|/|$)", stripped), (
            f"frontend image copies the whole build context: {stripped!r}. "
            "Copy explicit paths only."
        )


@pytest.mark.security
def test_frontend_install_is_frozen_lockfile(frontend_dockerfile: str) -> None:
    # A non-frozen `pnpm install` would let the image drift from the committed
    # lockfile — a supply-chain surface. Corepack provisions the pinned pnpm.
    assert "corepack" in frontend_dockerfile.lower(), (
        "frontend image must provision pnpm via Corepack (pinned packageManager)"
    )
    assert re.search(r"pnpm install\s+--frozen-lockfile", frontend_dockerfile), (
        "frontend image must run `pnpm install --frozen-lockfile`"
    )


@pytest.mark.security
def test_frontend_uses_standalone_output(frontend_dockerfile: str, next_config: str) -> None:
    # The runtime stage must ship Next's standalone server (no source tree, no
    # dev node_modules, no pnpm at runtime). That requires both the build-time
    # config opt-in AND the runtime COPY of the traced tree.
    assert re.search(r'output:\s*["\']standalone["\']', next_config), (
        "next.config.ts must set output: 'standalone' for the container image"
    )
    assert ".next/standalone" in frontend_dockerfile, (
        "frontend runtime stage does not copy Next's .next/standalone output"
    )
    assert ".next/static" in frontend_dockerfile, (
        "frontend runtime stage does not copy .next/static (the tracer omits it)"
    )


def test_frontend_cmd_runs_standalone_server(frontend_dockerfile: str) -> None:
    flat = re.sub(r"\\\s*\n", " ", frontend_dockerfile)
    cmd = next((ln for ln in flat.splitlines() if ln.strip().startswith("CMD")), "")
    assert "server.js" in cmd, "frontend CMD does not launch the standalone server.js"
    assert "HEALTHCHECK" in frontend_dockerfile, "frontend Dockerfile has no HEALTHCHECK"


@pytest.mark.security
def test_frontend_dockerignore_excludes_secrets_and_state(frontend_dockerignore: str) -> None:
    entries = {line.strip() for line in frontend_dockerignore.splitlines() if line.strip()}
    required = {
        ".env",
        ".env.*",
        "*.pem",
        "*.key",
        "node_modules/",
        ".next/",
        ".git/",
    }
    missing = sorted(required - entries)
    assert not missing, f"frontend/.dockerignore is missing secret/state exclusions: {missing}"
