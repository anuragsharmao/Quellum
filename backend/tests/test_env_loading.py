"""Regression: keys in backend/.env must reach Settings without exporting them by hand."""
import subprocess, sys, textwrap
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]


def test_dotenv_is_loaded(tmp_path):
    envfile = tmp_path / ".env"
    envfile.write_text("GROQ_API_KEY=test-key-123   # inline comment\n")
    code = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(BACKEND)!r})
        os.environ.pop("GROQ_API_KEY", None)
        from dotenv import load_dotenv
        load_dotenv({str(envfile)!r})
        from app.config import get_settings
        assert get_settings().groq_api_key == "test-key-123", get_settings().groq_api_key
    """)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
