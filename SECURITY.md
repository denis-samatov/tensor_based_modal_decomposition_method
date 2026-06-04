# Security Policy

## Supported Versions

This repository does not currently publish a formal support matrix. Security fixes should target the default branch unless maintainers define release branches.

## Reporting a Vulnerability

Do not open a public issue that contains secrets, credentials, private dataset details, or exploit instructions.

If private reporting is needed, contact the repository owner through the GitHub account associated with the project. Include:

- affected files or components;
- a concise description of the issue;
- reproduction steps when safe to share;
- impact and suggested mitigation.

## Sensitive Data

Do not commit:

- `.env` files;
- API keys, tokens, passwords, or certificates;
- private datasets or simulator outputs;
- trained models that embed private data;
- local absolute paths in stable docs or examples.

Use `.env.example` for non-sensitive configuration documentation only.
