"""Run the retention cleanup once and exit:  python -m app.cleanup_cli
Use this from an external scheduler (cron, systemd timer, Render/Railway/Fly cron job, GitHub Actions on a VM, Kubernetes CronJob).
Safe to run concurrently with the server and safe to retry."""
import json
import logging
import sys

from .config import db_path, get_settings
from .services.cleanup import run_cleanup
from .services.db import Database

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    s = get_settings()
    stats = run_cleanup(Database(db_path(s)), s)
    print(json.dumps(stats))
    sys.exit(1 if stats["errors"] else 0)
