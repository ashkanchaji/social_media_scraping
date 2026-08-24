#!/usr/bin/env bash
# One pass over every platform scraper, non-interactive.
#
# Schedule it every 10 minutes with cron (crontab -e):
#   */10 * * * * /home/achaji2563/Desktop/Codes/Social\ Media\ Scraping/run-scrapers.sh >> /home/achaji2563/Desktop/Codes/Social\ Media\ Scraping/run-scrapers.log 2>&1
#
# All configuration (discovery mode, time window, credentials) comes from .env,
# which pipeline_utils loads on import. Keep MAX_PAST_MINUTES larger than the
# cron interval so nothing falls between two runs; the dedup index drops the
# overlap.
set -u
cd "$(dirname "$0")" || exit 1

# A run that outlives the next cron tick must not be doubled up: the second
# invocation cannot take the lock and exits.
exec 9>".run-scrapers.lock"
flock -n 9 || { echo "$(date -Is) previous run still in progress, skipping"; exit 0; }

PY="$PWD/.venv/bin/python"
[ -x "$PY" ] || PY=python3

for scraper in twtr yt tel reddit tiktok truth; do
    echo "$(date -Is) === ${scraper}-scraper.py ==="
    # </dev/null keeps every scraper on its non-interactive path even if cron
    # ever hands it a terminal.
    "$PY" "${scraper}-scraper.py" </dev/null || echo "$(date -Is) FAILED: ${scraper}-scraper.py" >&2
done
echo "$(date -Is) === run complete ==="
