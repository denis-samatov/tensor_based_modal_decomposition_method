# Environment Variables

## Purpose
Documents the environment variables used by the project.

## Audience
Developers configuring their local environment.

## Summary
The TBMD project is a numerical library and does not require extensive environment variable configuration. 

## Details
There are no mandatory environment variables for the core library.

If you are running the Digital Twin examples that require external datasets, you may optionally specify local dataset paths via an `.env` file to avoid hardcoding paths in scripts.

Example `.env` format:
```env
BRUGGE_DATA_PATH=/path/to/local/data/brugge/
```
*(This is purely conventional for scripts and is not parsed by the core `TBMD` library).*

## Validation
Ensure `.env` files are not tracked by git.
