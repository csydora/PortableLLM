"""Start/stop only this project's background server."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import psutil

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / ".server.json"
RUN = ROOT / "run.py"
PYTHON = Path(sys.executable).resolve()

def owned_process():
    if not STATE.exists():
        return None
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
        process = psutil.Process(state["pid"])
        if (Path(process.exe()).resolve() == PYTHON
                and str(RUN) in process.cmdline()
                and abs(process.create_time() - state["created"]) < 0.01):
            return process
    except (psutil.Error, ValueError, KeyError, OSError):
        pass
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "stop"])
    action = parser.parse_args().action
    process = owned_process()
    if action == "stop":
        if process:
            process.terminate()
            process.wait(timeout=15)
            print("PortableLLM stopped.")
        else:
            print("No matching PortableLLM process is running.")
        STATE.unlink(missing_ok=True)
        return

    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    host = "127.0.0.1" if config["host"] == "0.0.0.0" else config["host"]
    url = f'http://{host}:{config["port"]}'
    if not process:
        import socket
        with socket.socket() as probe:
            if probe.connect_ex((host, config["port"])) == 0:
                raise RuntimeError(f"Port {config['port']} is occupied. No process was stopped.")
        logs = ROOT / "logs"
        logs.mkdir(exist_ok=True)
        with (logs / "server.out.log").open("w", encoding="utf-8") as out, (logs / "server.err.log").open("w", encoding="utf-8") as err:
            child = subprocess.Popen(
                [str(PYTHON), "-u", str(RUN)], cwd=ROOT,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        process = psutil.Process(child.pid)
        STATE.write_text(json.dumps({"pid": child.pid, "created": process.create_time()}), encoding="utf-8")
    for _ in range(60):
        try:
            with urlopen(url + "/", timeout=1) as response:
                if json.load(response).get("status") == "Running":
                    print(f"PortableLLM ready: {url}/docs")
                    return
        except OSError:
            pass
        if owned_process() is None:
            raise RuntimeError("Server exited. See logs/server.err.log.")
        time.sleep(0.5)
    raise RuntimeError(f"Server is still loading; see logs/server.err.log and {url}/docs.")


if __name__ == "__main__":
    main()

