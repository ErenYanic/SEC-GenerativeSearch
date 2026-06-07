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
from typing import Any

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOCKERFILE_API = _REPO_ROOT / "deploy" / "Dockerfile.api"
_DOCKERFILE_FRONTEND = _REPO_ROOT / "deploy" / "Dockerfile.frontend"
_ENTRYPOINT = _REPO_ROOT / "deploy" / "docker-entrypoint.sh"
_DOCKERIGNORE = _REPO_ROOT / ".dockerignore"
_FRONTEND_DOCKERIGNORE = _REPO_ROOT / "frontend" / ".dockerignore"
_NEXT_CONFIG = _REPO_ROOT / "frontend" / "next.config.ts"
_COMPOSE = _REPO_ROOT / "deploy" / "docker-compose.yml"
_NGINX_CONF = _REPO_ROOT / "deploy" / "nginx" / "nginx.conf"
_GITIGNORE = _REPO_ROOT / ".gitignore"
_CLOUD_API = _REPO_ROOT / "deploy" / "cloud" / "api-service.yaml"
_CLOUD_FRONTEND = _REPO_ROOT / "deploy" / "cloud" / "frontend-service.yaml"
_CLOUD_JOB = _REPO_ROOT / "deploy" / "cloud" / "demo-reset-job.yaml"

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


# ==========================================================================
# Compose + nginx portable stack (deploy/docker-compose.yml, deploy/nginx/
# nginx.conf) — the supply-chain + runtime contract for the whole deployment, including
# the in-process TaskManager contract and the edge-proxy contract. The compose file is
# the single source of truth for the deployment's supply chain (digest-pinned bases)
# and runtime configuration (replica count, env knobs, volume mounts); the nginx
# conf is the single source of truth for the edge proxy's routing and TLS contract.
# ===========================================================================


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compose_text() -> str:
    return _COMPOSE.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def nginx_conf() -> str:
    return _NGINX_CONF.read_text(encoding="utf-8")


# A secret-bearing env knob assigned a literal value in the compose file is a
# leak. A `${VAR}` interpolation or a `*_FILE` path indirection is fine.
_SECRET_ENV = (
    "API_KEY",
    "API_ADMIN_KEY",
    "API_AUTH_PEPPER",
    "DB_ENCRYPTION_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "HUGGING_FACE_TOKEN",
)


def test_compose_artifacts_exist() -> None:
    assert _COMPOSE.is_file(), "deploy/docker-compose.yml is missing"
    assert _NGINX_CONF.is_file(), "deploy/nginx/nginx.conf is missing"


def _services(compose: dict[str, Any]) -> dict[str, Any]:
    services = compose.get("services", {})
    assert {"api", "frontend", "nginx"} <= set(services), (
        f"compose must define api + frontend + nginx services; got {sorted(services)}"
    )
    return services


# ---------------------------------------------------------------------------
# Supply chain: the third-party (nginx) base image is digest-pinned.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_compose_nginx_image_is_digest_pinned(compose: dict[str, Any]) -> None:
    image = _services(compose)["nginx"].get("image", "")
    assert _DIGEST_RE.search(image), (
        f"nginx image is not digest-pinned (@sha256:...): {image!r}. "
        "A floating :tag is a silent-substitution / supply-chain vector."
    )


# ---------------------------------------------------------------------------
# In-process TaskManager: exactly one api replica.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_compose_api_is_single_replica(compose: dict[str, Any]) -> None:
    api = _services(compose)["api"]
    replicas = api.get("deploy", {}).get("replicas", 1)
    assert replicas == 1, (
        f"api must run exactly one replica (in-process TaskManager mints task "
        f"IDs in-process; a second replica orphans them); got replicas={replicas}"
    )
    # `scale:` is the deprecated knob for the same thing — pin it too.
    assert "scale" not in api or api["scale"] == 1, "api must not scale beyond one instance"


# ---------------------------------------------------------------------------
# Only the edge publishes host ports; api + frontend stay internal.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_compose_only_nginx_publishes_ports(compose: dict[str, Any]) -> None:
    services = _services(compose)
    for name in ("api", "frontend"):
        assert not services[name].get("ports"), (
            f"{name} publishes host ports — the browser must reach the backend only "
            "through the Next proxy; only nginx should publish ports."
        )
    assert services["nginx"].get("ports"), "nginx publishes no host ports — nothing is reachable"


# ---------------------------------------------------------------------------
# No baked secret; secret-bearing knobs only via *_FILE / interpolation.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_compose_bakes_no_secret(compose_text: str) -> None:
    lowered = compose_text.lower()
    assert not re.search(r"\bsk-[a-z0-9]{8,}", lowered), (
        "docker-compose.yml contains an sk- API-key-shaped token"
    )
    for needle in ("bearer ", "authorization:", "private key"):
        assert needle not in lowered, f"compose contains secret-shaped material: {needle!r}"
    # No secret-bearing env knob may be ASSIGNED a literal value. Allow the
    # `*_FILE` path indirection (DB_ENCRYPTION_KEY_FILE: /run/secrets/...) and a
    # `${VAR}` interpolation; a bare `NAME: <value>` / `NAME=<value>` is a leak.
    for name in _SECRET_ENV:
        bad = re.search(rf"^\s*-?\s*{name}\s*[:=]\s*(?!\s|$|\$|/run/secrets)\S", compose_text, re.M)
        assert not bad, (
            f"compose assigns a literal value to {name} — supply it at runtime via "
            "env_file / a mounted *_FILE secret, never inline."
        )


@pytest.mark.security
def test_compose_api_uses_secret_file_indirection(compose: dict[str, Any]) -> None:
    api = _services(compose)["api"]
    env = api.get("environment", {}) or {}
    # The encryption key + auth pepper must arrive via a mounted file, not a
    # literal env value — so they live in /run/secrets, never in the image or
    # the process env table as cleartext.
    for knob in ("DB_ENCRYPTION_KEY_FILE", "API_AUTH_PEPPER_FILE"):
        value = env.get(knob, "")
        assert value.startswith("/run/secrets/"), (
            f"api {knob} must point at a mounted Docker-secret under /run/secrets/; got {value!r}"
        )
    # And those files must be wired as Docker secrets (read-only /run/secrets
    # mount that fails loudly when the source file is absent).
    secret_refs = {s if isinstance(s, str) else s.get("source") for s in (api.get("secrets") or [])}
    assert {"db_encryption_key", "api_auth_pepper"} <= secret_refs, (
        f"api must mount db_encryption_key + api_auth_pepper as Docker secrets; got {secret_refs}"
    )


@pytest.mark.security
def test_compose_top_level_secrets_are_file_sourced(compose: dict[str, Any]) -> None:
    secrets = compose.get("secrets", {})
    for name in ("db_encryption_key", "api_auth_pepper"):
        spec = secrets.get(name, {})
        assert "file" in spec, f"secret {name} must be file-sourced (operator-supplied), not inline"
        assert "environment" not in spec, (
            f"secret {name} must not be sourced from an env var (would bake into the process table)"
        )


@pytest.mark.security
def test_compose_frontend_is_keyless(compose: dict[str, Any]) -> None:
    fe = _services(compose)["frontend"]
    env = fe.get("environment", {}) or {}
    # The only runtime knob is the server-side backend base URL. No key, and
    # crucially no NEXT_PUBLIC_* (that would ship to the browser).
    for key in env:
        assert not key.startswith("NEXT_PUBLIC_"), (
            f"frontend sets {key} — a NEXT_PUBLIC_* var reaches the browser bundle"
        )
    assert "SEC_API_BASE_URL" in env, "frontend must point at the backend via SEC_API_BASE_URL"
    # The base URL is server-side; it must NOT be advertised through a public var.
    assert "NEXT_PUBLIC_SEC_API_BASE_URL" not in env, (
        "SEC_API_BASE_URL must stay server-side, never NEXT_PUBLIC_*"
    )


@pytest.mark.security
def test_compose_api_persists_data_on_a_volume(compose: dict[str, Any]) -> None:
    api = _services(compose)["api"]
    targets = []
    for vol in api.get("volumes", []) or []:
        if isinstance(vol, str):
            targets.append(vol.split(":")[1] if ":" in vol else vol)
        else:
            targets.append(vol.get("target"))
    assert "/app/data" in targets, (
        "api must mount a durable volume at /app/data — task_history is the only "
        "crash-durable ingest record (§4.7.quinquies)"
    )


# ---------------------------------------------------------------------------
# .gitignore re-ignores the operator secret / TLS material under deploy/.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_gitignore_reignores_deploy_secrets() -> None:
    gitignore = _GITIGNORE.read_text(encoding="utf-8")
    carve_out = gitignore.index("!deploy/")
    for pattern in ("deploy/secrets/", "deploy/certs/"):
        idx = gitignore.find(pattern)
        assert idx != -1, f".gitignore must re-ignore {pattern} (operator secret/TLS material)"
        assert idx > carve_out, (
            f"{pattern} re-ignore must come AFTER the !deploy/ carve-out or it has no effect"
        )


# ==========================================================================
# nginx reverse proxy — routing + TLS only; owns NO CSP.
# ==========================================================================

# Security / CSP headers nginx MUST NOT inject — the Next middleware owns the
# full set (a second, un-nonced policy here would fight or weaken it).
_FORBIDDEN_NGINX_HEADERS = (
    "content-security-policy",
    "strict-transport-security",
    "x-frame-options",
    "x-content-type-options",
    "referrer-policy",
    "permissions-policy",
)


def _nginx_add_headers(nginx_conf: str) -> list[str]:
    """Lower-cased header names from every (uncommented) add_header directive."""
    names: list[str] = []
    for raw in nginx_conf.splitlines():
        line = raw.strip()
        if line.startswith("#"):
            continue
        match = re.match(r"add_header\s+([A-Za-z0-9-]+)", line)
        if match:
            names.append(match.group(1).lower())
    return names


def test_nginx_conf_exists() -> None:
    assert _NGINX_CONF.is_file(), "deploy/nginx/nginx.conf is missing"


@pytest.mark.security
def test_nginx_sets_no_csp_or_security_headers(nginx_conf: str) -> None:
    emitted = _nginx_add_headers(nginx_conf)
    leaked = sorted(set(emitted) & set(_FORBIDDEN_NGINX_HEADERS))
    assert not leaked, (
        f"nginx injects security/CSP header(s) {leaked} — the Next.js middleware is the "
        "single source of truth for CSP + the security-header set. nginx does routing only."
    )


@pytest.mark.security
def test_nginx_admin_proxy_routes_to_frontend_before_api(nginx_conf: str) -> None:
    # The browser reaches the backend ONLY through the Next admin proxy at
    # /api/admin/*. That route lives on the frontend (it injects the server-held
    # keys), so its location MUST proxy to the frontend AND be declared before the
    # bare /api/ location so longest-prefix-wins keeps it ahead.
    admin_idx = nginx_conf.find("location /api/admin/")
    api_idx = nginx_conf.find("location /api/ ")
    assert admin_idx != -1, "nginx has no `location /api/admin/` — SPA admin proxy unreachable"
    assert api_idx != -1, "nginx has no `location /api/` for direct API consumers"
    assert admin_idx < api_idx, (
        "`location /api/admin/` must precede `location /api/` so the admin proxy wins"
    )
    # The admin-proxy block targets the frontend upstream; the bare /api/ block
    # targets the api upstream.
    admin_block = nginx_conf[admin_idx:api_idx]
    assert "proxy_pass http://frontend" in admin_block, (
        "`location /api/admin/` must proxy to the frontend (the key-injecting Next proxy)"
    )
    api_block = nginx_conf[api_idx : api_idx + 600]
    assert "proxy_pass http://api" in api_block, "`location /api/` must proxy to the api upstream"


@pytest.mark.security
def test_nginx_websocket_upgrade_to_api(nginx_conf: str) -> None:
    ws_idx = nginx_conf.find("location /ws/")
    assert ws_idx != -1, "nginx has no `location /ws/` for the ingest-progress WebSocket"
    ws_block = nginx_conf[ws_idx : ws_idx + 600]
    assert "proxy_pass http://api" in ws_block, "`location /ws/` must proxy to the api upstream"
    assert re.search(r"proxy_set_header\s+Upgrade\s+\$http_upgrade", ws_block), (
        "WebSocket location must forward the Upgrade header"
    )
    assert re.search(r"proxy_set_header\s+Connection\s+\$connection_upgrade", ws_block), (
        "WebSocket location must set Connection: upgrade"
    )
    # The backend's WS authorisation surface is the Origin allow-list — nginx
    # must forward Origin intact for that check to run.
    assert re.search(r"proxy_set_header\s+Origin\s+\$http_origin", ws_block), (
        "WebSocket location must forward the Origin header (backend WS auth surface)"
    )


def test_nginx_terminates_tls(nginx_conf: str) -> None:
    assert "listen 443 ssl" in nginx_conf, "nginx does not terminate TLS on :443"
    assert "ssl_certificate" in nginx_conf, "nginx has no TLS certificate directive"


@pytest.mark.security
def test_nginx_no_baked_secret(nginx_conf: str) -> None:
    lowered = nginx_conf.lower()
    assert not re.search(r"\bsk-[a-z0-9]{8,}", lowered), "nginx.conf contains an sk- API-key token"
    assert "private key" not in lowered, "nginx.conf contains secret-shaped material"


# ==========================================================================
# GCP Cloud Run manifests (deploy/cloud/{api,frontend}-service.yaml,
# demo-reset-job.yaml) — the Cloud Run counterparts of the Compose stack and
# carry the same load-bearing contracts, expressed in Knative annotations:
#
#   - the in-process TaskManager single-instance contract (maxScale=1, no
#     scale-to-zero, no CPU throttling between requests);
#   - Secret Manager indirection for every secret-bearing knob (the Cloud Run
#     analogue of the Compose *_FILE / Docker-secret mounts);
#   - a keyless, GFE-TLS frontend service whose only env knob is the server-side
#     SEC_API_BASE_URL (no NEXT_PUBLIC_*, no secret);
#   - an internal-ingress API the browser can never reach directly.
#
# Like every other locker here, the assertions are on tracked, CI-visible
# files and need no gcloud / network / Docker daemon.
# ===========================================================================

_KNATIVE_SERVICE_KIND = "Service"
_CLOUD_RUN_JOB_KIND = "Job"


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _service_container(doc: dict[str, Any]) -> dict[str, Any]:
    """The first container of a Knative ``Service`` revision template."""
    containers = doc["spec"]["template"]["spec"]["containers"]
    assert containers, "Knative Service defines no container"
    return containers[0]


def _job_container(doc: dict[str, Any]) -> dict[str, Any]:
    """The first container of a Cloud Run ``Job`` task template.

    The nesting is Job → ExecutionTemplate (``spec.template``) → TaskTemplate
    (``.spec.template``) → ``.spec.containers``.
    """
    containers = doc["spec"]["template"]["spec"]["template"]["spec"]["containers"]
    assert containers, "Cloud Run Job defines no container"
    return containers[0]


def _template_annotations(doc: dict[str, Any]) -> dict[str, str]:
    return doc["spec"]["template"]["metadata"].get("annotations", {}) or {}


def _env_by_name(container: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["name"]: entry for entry in (container.get("env") or [])}


def _uses_secret_manager(entry: dict[str, Any]) -> bool:
    """True iff the env entry resolves from Secret Manager (no inline value)."""
    if "value" in entry:
        return False
    return "secretKeyRef" in (entry.get("valueFrom") or {})


@pytest.fixture(scope="module")
def cloud_api() -> dict[str, Any]:
    return _load_yaml(_CLOUD_API)


@pytest.fixture(scope="module")
def cloud_frontend() -> dict[str, Any]:
    return _load_yaml(_CLOUD_FRONTEND)


@pytest.fixture(scope="module")
def cloud_job() -> dict[str, Any]:
    return _load_yaml(_CLOUD_JOB)


def test_cloud_artifacts_exist() -> None:
    assert _CLOUD_API.is_file(), "deploy/cloud/api-service.yaml is missing"
    assert _CLOUD_FRONTEND.is_file(), "deploy/cloud/frontend-service.yaml is missing"
    assert _CLOUD_JOB.is_file(), "deploy/cloud/demo-reset-job.yaml is missing"


def test_cloud_services_have_expected_kinds(
    cloud_api: dict[str, Any], cloud_frontend: dict[str, Any], cloud_job: dict[str, Any]
) -> None:
    assert cloud_api["kind"] == _KNATIVE_SERVICE_KIND, "api-service.yaml must be a Knative Service"
    assert cloud_frontend["kind"] == _KNATIVE_SERVICE_KIND, (
        "frontend-service.yaml must be a Knative Service"
    )
    assert cloud_job["kind"] == _CLOUD_RUN_JOB_KIND, "demo-reset-job.yaml must be a Cloud Run Job"


# ---------------------------------------------------------------------------
# In-process TaskManager: exactly one API instance, never scaled to zero.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_cloud_api_is_single_instance(cloud_api: dict[str, Any]) -> None:
    ann = _template_annotations(cloud_api)
    max_scale = ann.get("autoscaling.knative.dev/maxScale")
    assert max_scale == "1", (
        f"api maxScale must be '1' (the in-process TaskManager mints task IDs "
        f"in-process; a second instance orphans them); got {max_scale!r}"
    )


@pytest.mark.security
def test_cloud_api_never_scales_to_zero(cloud_api: dict[str, Any]) -> None:
    # Scale-to-zero or CPU-throttling between requests kills the daemon-thread
    # ingest workers mid-flight (DEPLOYMENT §4.7.quinquies).
    ann = _template_annotations(cloud_api)
    min_scale = ann.get("autoscaling.knative.dev/minScale")
    assert min_scale is not None and int(min_scale) >= 1, (
        f"api minScale must be >= 1 — scale-to-zero kills in-flight ingest "
        f"workers; got {min_scale!r}"
    )
    throttle = ann.get("run.googleapis.com/cpu-throttling")
    assert throttle == "false", (
        f"api cpu-throttling must be 'false' so background worker threads keep "
        f"running between requests; got {throttle!r}"
    )


@pytest.mark.security
def test_cloud_api_ingress_is_internal(cloud_api: dict[str, Any]) -> None:
    # The browser must reach the backend only through the keyless Next admin
    # proxy on the frontend service — never the API directly.
    ingress = cloud_api["metadata"].get("annotations", {}).get("run.googleapis.com/ingress")
    assert ingress == "internal", (
        f"api ingress must be 'internal' (browser reaches it only via the Next "
        f"proxy on the frontend); got {ingress!r}"
    )


# ---------------------------------------------------------------------------
# No baked secret; secret-bearing knobs only via Secret Manager.
# ---------------------------------------------------------------------------


@pytest.mark.security
@pytest.mark.parametrize("path", [_CLOUD_API, _CLOUD_FRONTEND, _CLOUD_JOB], ids=lambda p: p.name)
def test_cloud_manifest_bakes_no_secret(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lowered = text.lower()
    assert not re.search(r"\bsk-[a-z0-9]{8,}", lowered), (
        f"{path.name} contains an sk- API-key-shaped token"
    )
    for needle in ("bearer ", "authorization:", "private key", "begin certificate"):
        assert needle not in lowered, f"{path.name} contains secret-shaped material: {needle!r}"
    # No secret-bearing knob may be ASSIGNED a literal `value:`. The Secret
    # Manager indirection (valueFrom.secretKeyRef) carries no literal here, so a
    # `<NAME>` ... `value: <x>` pairing on adjacent lines is the leak shape.
    for name in _SECRET_ENV:
        bad = re.search(rf"name:\s*{name}\b[^\n]*\n\s*value:\s*\S", text)
        assert not bad, (
            f"{path.name} assigns a literal `value:` to {name} — resolve it from "
            "Secret Manager via valueFrom.secretKeyRef, never inline."
        )


@pytest.mark.security
def test_cloud_api_secret_knobs_use_secret_manager(cloud_api: dict[str, Any]) -> None:
    env = _env_by_name(_service_container(cloud_api))
    present_secrets = [name for name in _SECRET_ENV if name in env]
    # Sanity: the cloud API actually configures the encryption key + pepper, so
    # the assertion below is never vacuous.
    assert {"DB_ENCRYPTION_KEY", "API_AUTH_PEPPER"} <= set(present_secrets), (
        f"cloud API must configure DB_ENCRYPTION_KEY + API_AUTH_PEPPER; got {present_secrets}"
    )
    for name in present_secrets:
        assert _uses_secret_manager(env[name]), (
            f"api {name} must resolve from Secret Manager (valueFrom.secretKeyRef), "
            f"never an inline value; got {env[name]!r}"
        )


@pytest.mark.security
def test_cloud_api_persists_data_via_gcsfuse(cloud_api: dict[str, Any]) -> None:
    container = _service_container(cloud_api)
    mounts = {m.get("mountPath") for m in (container.get("volumeMounts") or [])}
    assert "/app/data" in mounts, (
        "api must mount a durable volume at /app/data — task_history is the only "
        "crash-durable ingest record (§4.7.quinquies)"
    )
    volumes = cloud_api["spec"]["template"]["spec"].get("volumes") or []
    drivers = {v.get("csi", {}).get("driver") for v in volumes}
    assert "gcsfuse.run.googleapis.com" in drivers, (
        "api /app/data must be backed by the GCS FUSE CSI driver on Cloud Run "
        f"(no persistent local disk); got volume drivers {drivers}"
    )


# ---------------------------------------------------------------------------
# Keyless, GFE-TLS frontend service.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_cloud_frontend_is_keyless(cloud_frontend: dict[str, Any]) -> None:
    container = _service_container(cloud_frontend)
    env = _env_by_name(container)
    for name, entry in env.items():
        assert not name.startswith("NEXT_PUBLIC_"), (
            f"frontend sets {name} — a NEXT_PUBLIC_* var reaches the browser bundle"
        )
        assert name not in _SECRET_ENV, f"frontend env carries a secret-bearing knob {name}"
        assert "valueFrom" not in entry, (
            f"frontend env {name} resolves from a secret store — the SPA image is keyless"
        )
    assert "SEC_API_BASE_URL" in env, "frontend must point at the backend via SEC_API_BASE_URL"
    assert "NEXT_PUBLIC_SEC_API_BASE_URL" not in env, (
        "SEC_API_BASE_URL must stay server-side, never NEXT_PUBLIC_*"
    )


@pytest.mark.security
def test_cloud_frontend_is_public_gfe_tls(cloud_frontend: dict[str, Any]) -> None:
    # GFE terminates TLS and serves a managed certificate for a public service;
    # the manifest defines NO TLS material of its own, and the service is
    # publicly reachable (ingress: all, or unset which defaults to all).
    ingress = (
        cloud_frontend["metadata"].get("annotations", {}).get("run.googleapis.com/ingress", "all")
    )
    assert ingress == "all", (
        f"frontend must be publicly reachable (ingress 'all') so GFE fronts it "
        f"with managed TLS; got {ingress!r}"
    )
    text = _CLOUD_FRONTEND.read_text(encoding="utf-8").lower()
    for needle in ("ssl_certificate", "tls_cert", "443", "begin private key"):
        assert needle not in text, (
            f"frontend manifest defines TLS material ({needle!r}); GFE owns TLS — "
            "the service must not terminate it"
        )


# ---------------------------------------------------------------------------
# Demo-reset Cloud Run Job.
# ---------------------------------------------------------------------------


@pytest.mark.security
def test_cloud_demo_reset_is_a_job(cloud_job: dict[str, Any]) -> None:
    # A Job runs to completion per trigger — never a long-lived Service that
    # would hold the destructive `clear` surface open.
    assert cloud_job["kind"] == _CLOUD_RUN_JOB_KIND, "demo-reset must be a Cloud Run Job"
    container = _job_container(cloud_job)
    # It invokes the CLI (which bypasses API_DEMO_MODE) — the only reset path,
    # since the API blocks `clear` under demo mode.
    args = container.get("args") or []
    assert args[:3] == ["sec-rag", "manage", "clear"], (
        f"demo-reset Job must run `sec-rag manage clear` (the CLI reset path that "
        f"bypasses demo mode); got args {args}"
    )
    # `args` only, no `command`: the image ENTRYPOINT (the gosu drop) stays in
    # force, so the reset runs as the unprivileged appuser, not root.
    assert "command" not in container, (
        "demo-reset Job must not override `command` — keep the image ENTRYPOINT "
        "(docker-entrypoint.sh) so the gosu non-root drop still runs"
    )


@pytest.mark.security
def test_cloud_demo_reset_uses_secret_manager(cloud_job: dict[str, Any]) -> None:
    env = _env_by_name(_job_container(cloud_job))
    assert "DB_ENCRYPTION_KEY" in env, (
        "demo-reset Job needs DB_ENCRYPTION_KEY to open the SQLCipher store"
    )
    assert _uses_secret_manager(env["DB_ENCRYPTION_KEY"]), (
        "demo-reset Job DB_ENCRYPTION_KEY must resolve from Secret Manager "
        "(valueFrom.secretKeyRef), never an inline value"
    )
