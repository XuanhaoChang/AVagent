# Security Policy

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability or exposed
credential. Use GitHub's private vulnerability reporting for this repository.
Include affected files or versions, reproduction steps, impact, and any known
mitigation. Do not include real API keys, private media, or private datasets.

## Credential handling

Store credentials only in local environment variables or `.env.local`. AVAgent
ignores `.env.local`, datasets, outputs, checkpoints, caches, and model weights.
If a credential is committed or posted publicly, revoke it before removing it
from Git history.
