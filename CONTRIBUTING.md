# Contributing to DeepVariant Linux ARM64

Technical workflows, coding standards, and command reference for contributors.
For agent behavioral rules, see [AGENTS.md](AGENTS.md).

## Instant Commands

| Command | Purpose |
|---------|---------|
| `make setup_dev` | Install dev dependencies via uv |
| `make lint` | Format and lint Python with ruff |
| `make type_check` | Run pyright type checking |
| `make test` | Run tests with pytest |
| `make validate` | Full validation (lint + type_check + test) |
| `make quick_validate` | Fast validation (lint + type_check) |
| `make docker_build` | Build ARM64 Docker image |
| `make benchmark` | Run chr20 benchmark (requires ARM64 + Docker) |

## Development Workflow

### Environment Setup

```bash
# Install uv (if not already available)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dev dependencies
make setup_dev

# Docker build (requires buildx + QEMU or native ARM64)
make docker_build
```

### Pre-commit Checklist

1. `make validate` — must pass
2. Update `CHANGELOG.md` for non-trivial changes
3. Commit with Conventional Commits format

### Docker Build & Test

The primary deliverable is `docker/Dockerfile.arm64`. Build and test flow:

```bash
# Build
make docker_build

# Benchmark (on ARM64 host)
make benchmark

# Accuracy validation
bash scripts/validate_accuracy.sh
```

## Coding Standards

### Python (scripts/ and deepvariant/*.py)

- **Formatter/Linter:** ruff (config in `pyproject.toml`)
- **Type checker:** pyright (basic mode)
- **Style:** Single quotes, 100 char line length
- **Imports:** Standard library > third-party > local, sorted by isort
- **Tests:** Files named `*_test.py`, functions named `test_*`

### Shell Scripts

- Use `set -euo pipefail` at the top
- Quote all variables: `"${VAR}"`
- Use `#!/usr/bin/env bash` shebang

### C++ (deepvariant/)

- Follow upstream google/deepvariant style (Google C++ Style Guide)
- ARM64-specific changes clearly commented with `// ARM64:`

## Code Review & PR Guidelines

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add INT8 ONNX quantization for ARM64
fix: resolve SIGILL on AmpereOne with OneDNN disabled
docs: update benchmark results for Graviton4
perf: optimize make_examples with jemalloc allocator
```

### PR Requirements

- Clear description of changes and motivation
- `make validate` passes
- Benchmark results included for performance changes
- Accuracy validation for inference changes

## Project Structure

```
deepvariant-linux-arm64/
├── CLAUDE.md                    # Agent entry point (@AGENTS.md)
├── AGENTS.md                    # Agent behavioral rules
├── CONTRIBUTING.md              # This file
├── AGENT_LEARNINGS.md           # Accumulated agent knowledge
├── Makefile                     # Development commands
├── pyproject.toml               # Python tooling config
├── docker/
│   ├── Dockerfile.arm64         # ARM64 Docker build
│   └── Dockerfile.arm64.runtime # ARM64 runtime image
├── scripts/
│   └── build/
│       └── settings_arm64.sh    # ARM64 build flags
├── scripts/                     # Benchmarks, conversion, run wrappers
├── deepvariant/                 # Upstream source + ARM64 modifications
├── docs/
│   └── architecture-arm64.md    # Full ARM64 architecture reference
└── .claude/
    ├── settings.json            # Marketplace plugin config
    ├── rules/                   # Core principles, context management (plugin-managed)
    └── skills/                  # Code review, codebase research, commit (plugin-managed)
```

## Documentation Hierarchy

- **AGENTS.md** — agent rules and quick reference (authority for behavior)
- **CONTRIBUTING.md** — technical workflows (authority for how-to)
- **docs/architecture-arm64.md** — full architecture, benchmarks, decisions (reference)
- **CHANGELOG.md** — release history
- **README.md** — project overview and quick start
