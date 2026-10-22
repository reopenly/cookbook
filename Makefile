.PHONY: check check-live secrets

check:          ## offline checks; no API key or paid calls
	uv run --locked scripts/check.py

check-live:     ## paid calls and eight persistent demo events; requires explicit project/dataset/model
	uv run --env-file .env --locked scripts/check.py --live

secrets:        ## inspect release files and all locally reachable Git history
	uv run --locked scripts/scan_secrets.py
