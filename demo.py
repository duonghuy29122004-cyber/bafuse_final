"""Launch the existing BaFuse Streamlit application."""
import os
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root / "src"), env.get("PYTHONPATH"))))
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(root / "app/final2_streamlit_dashboard.py"),
         "--server.address", "127.0.0.1", "--server.port", "8767",
         "--server.maxUploadSize", "1024", "--browser.gatherUsageStats", "false",
         *sys.argv[1:]], cwd=root, env=env,
    ))
