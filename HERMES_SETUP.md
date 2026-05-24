# Hermes Agent Setup for IVS News

**Status**: Active (as of May 24, 2026)

## Profile
- Name: `ivs-research`
- Model: Grok-4.3 (via xAI OAuth)
- Command to chat: `ivs-research chat`
- Persistent gateway running via launchd

## Key Cron Jobs Created
- `daily-ivs-content-monitor` (08:00 daily)
- `daily-competitor-tracker` (09:00 daily)
- `daily-pipeline-auditor` (10:00 daily)

## Useful Locations
- SOUL.md: `~/.hermes/profiles/ivs-research/SOUL.md`
- Output files: `~/.hermes/profiles/ivs-research/cron/output/`
- Provider list: `ivsnews-directory-providers.txt` (in project root)

## Start Commands
```bash
ivs-research chat
ivs-research status

