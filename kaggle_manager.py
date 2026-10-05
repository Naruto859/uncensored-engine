import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional

from models_registry import MODELS, DEFAULT_MODEL_ID

logger = logging.getLogger("kaggle_manager")

KAGGLE_VENV_BIN = Path("/root/kaggle-gpu/venv/bin/kaggle")
DEFAULT_KAGGLE_CREDS = Path("/root/.kaggle/kaggle.json")
KAGGLE_CONFIG_PATH = DEFAULT_KAGGLE_CREDS
BUNDLE_DIR = Path("/root/uncensored-manager/kernel_bundle")


class KaggleKernelManager:
    def __init__(self, config_callback=None):
        self.config_callback = config_callback
        self.active_model_id = DEFAULT_MODEL_ID
        self.tunnel_url = ""
        self.status = "idle"  # idle, initializing, downloading, loading_vram, tunnel_active, ready, error
        self.status_detail = "System initialized and waiting for launch."
        self.launch_time: Optional[float] = None
        self.kernel_slug = "samirandas22/hermes-uncensored-engine"
        self.tunnel_type = "cloudflare"  # cloudflare, ngrok
        self.ngrok_token = ""
        self.webhook_host = "http://164.68.114.240:8778"
        self.recent_logs = []
        self._polling_thread = None
        self._stop_polling = threading.Event()
        self.kaggle_proxy = "server"
        self.load_kaggle_credentials()

    def get_subprocess_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        proxy = getattr(self, "kaggle_proxy", "server")
        if proxy and proxy.lower() not in ("server", "direct", "none", ""):
            env["HTTP_PROXY"] = proxy
            env["HTTPS_PROXY"] = proxy
            env["http_proxy"] = proxy
            env["https_proxy"] = proxy
        else:
            for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"]:
                env.pop(k, None)
        return env

    def load_kaggle_credentials(self) -> Dict[str, Any]:
        username = "samirandas22"
        key_present = False
        if DEFAULT_KAGGLE_CREDS.exists():
            try:
                data = json.loads(DEFAULT_KAGGLE_CREDS.read_text(encoding="utf-8"))
                username = data.get("username", "samirandas22")
                key_present = bool(data.get("key"))
                self.kernel_slug = f"{username}/hermes-uncensored-engine"
            except Exception as e:
                logger.warning(f"Error loading kaggle credentials: {e}")
        return {
            "username": username,
            "key_present": key_present,
            "proxy": getattr(self, "kaggle_proxy", "server")
        }

    def log(self, message: str):
        timestamp = time.strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        logger.info(entry)
        self.recent_logs.append(entry)
        if len(self.recent_logs) > 200:
            self.recent_logs.pop(0)

    def set_status(self, status: str, detail: str = ""):
        self.status = status
        if detail:
            self.status_detail = detail
        self.log(f"Status changed -> {status.upper()}: {detail}")
        if self.config_callback:
            self.config_callback()

    def set_tunnel_url(self, url: str):
        self.tunnel_url = url.rstrip("/")
        self.log(f"Active Tunnel Upstream URL registered: {self.tunnel_url}")
        self.set_status("ready", f"Tunnel live: {self.tunnel_url}")

    def generate_kernel_script(self, model_id: str) -> str:
        model = MODELS.get(model_id, MODELS[DEFAULT_MODEL_ID])
        server_args_str = ", ".join([json.dumps(arg) for arg in model["server_args"]])
        model_name = model["name"]
        model_params = model["parameters"]
        model_quant = model["quant"]
        repo_id = model["repo_id"]
        filename = model["filename"]
        filesize_gb = model["filesize_gb"]
        model_key = model["id"]
        webhook_target = f"{self.webhook_host}/api/tunnel/register"

        ctx_disp = model.get("context_display", f"{model.get('context_length', 32768)} tokens")
        rope_disp = model.get("rope_scaling", "Native / None")
        kv_disp = model.get("kv_cache", "q4_0")

        script_code = f'''#!/usr/bin/env python3
# ==============================================================================
# Autonomous Uncensored AI Agent Kernel - Dual Tesla T4 / P100 Execution Engine
# Model: {model_name} ({model_params}, {model_quant})
# Context Window: {ctx_disp} | RoPE: {rope_disp} | KV: {kv_disp}
# Hugging Face: {repo_id} / {filename}
# ==============================================================================

import os
import sys
import time
import subprocess
import json
import urllib.request
import re

print("=" * 70)
print("[ENGINE] Booting Autonomous Uncensored AI Engine on Kaggle...")
print("=" * 70)

MODEL_NAME = {json.dumps(model_name)}
MODEL_REPO = {json.dumps(repo_id)}
MODEL_FILENAME = {json.dumps(filename)}
MODEL_FILESIZE = {filesize_gb}
MODEL_ID = {json.dumps(model_key)}
WEBHOOK_TARGET = {json.dumps(webhook_target)}

# Check GPU hardware
print("[GPU CHECK] Inspecting CUDA accelerators...")
try:
    smi = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT).decode()
    print(smi)
except Exception as e:
    print("[WARN] nvidia-smi failed:", e)

WORK = "/kaggle/working"
MODEL_DIR = "/tmp/model"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("/tmp/bin", exist_ok=True)

# Step 1: Install Python prerequisites
print("\\n[STEP 1] Installing huggingface_hub and requests...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "requests"], check=False)

# Step 2: Acquire llama.cpp binary (Prebuilt CUDA or fast source build)
LLAMA_SERVER = "/tmp/bin/llama-server"

def setup_llama_server():
    global LLAMA_SERVER
    print("\\n[STEP 2] Setting up llama-server with GPU support...")
    
    # Try downloading official release b11146 prebuilt for Ubuntu CUDA 12.8
    tar_url = "https://github.com/ggml-org/llama.cpp/releases/download/b11146/llama-b11146-bin-ubuntu-cuda-12.8-x64.tar.gz"
    cudart_url = "https://github.com/ggml-org/llama.cpp/releases/download/b11146/cudart-llama-b11146-bin-ubuntu-cuda-12.8-x64.tar.gz"
    
    try:
        print("[LLAMA] Attempting direct binary download from ggml-org release b11146...")
        subprocess.run(["curl", "-sL", "-o", "/tmp/llama.tar.gz", tar_url], check=True)
        subprocess.run(["curl", "-sL", "-o", "/tmp/cudart.tar.gz", cudart_url], check=True)
        
        subprocess.run(["tar", "-xzf", "/tmp/llama.tar.gz", "-C", "/tmp/bin"], check=True)
        subprocess.run(["tar", "-xzf", "/tmp/cudart.tar.gz", "-C", "/tmp/bin"], check=True)
        
        # Check if llama-server exists
        for root, dirs, files in os.walk("/tmp/bin"):
            if "llama-server" in files:
                LLAMA_SERVER = os.path.join(root, "llama-server")
                os.chmod(LLAMA_SERVER, 0o755)
                print(f"[LLAMA] Successfully installed pre-built llama-server at: {{LLAMA_SERVER}}")
                return True
    except Exception as e:
        print("[LLAMA] Direct prebuilt download encountered issue:", e)
        print("[LLAMA] Falling back to building llama.cpp from source with CUDA...")

    # Fallback: Git clone & CMake build
    os.chdir(WORK)
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git"], check=True)
    build_dir = os.path.join(WORK, "llama.cpp", "build")
    os.makedirs(build_dir, exist_ok=True)
    os.chdir(os.path.join(WORK, "llama.cpp"))
    
    # Configure with CUDA for Tesla T4 (arch 75)
    subprocess.run([
        "cmake", "-B", "build",
        "-DGGML_CUDA=ON",
        "-DCMAKE_CUDA_ARCHITECTURES=75",
        "-DCMAKE_BUILD_TYPE=Release"
    ], check=True)
    subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j4", "--target", "llama-server"], check=True)
    
    for candidate in [
        os.path.join(build_dir, "bin", "llama-server"),
        os.path.join(build_dir, "llama-server")
    ]:
        if os.path.exists(candidate):
            LLAMA_SERVER = candidate
            os.chmod(LLAMA_SERVER, 0o755)
            print(f"[LLAMA] Compiled llama-server at: {{LLAMA_SERVER}}")
            return True
            
    raise RuntimeError("Could not establish llama-server binary!")

setup_llama_server()

# Step 3: Download Model GGUF
print(f"\\n[STEP 3] Downloading GGUF model: {{MODEL_REPO}} / {{MODEL_FILENAME}}...")
from huggingface_hub import hf_hub_download

model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
if not os.path.exists(model_path):
    print(f"[MODEL] Fetching {{MODEL_FILENAME}} (~{{MODEL_FILESIZE}} GB)...")
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=MODEL_DIR
    )
    print(f"[MODEL] Model downloaded successfully to: {{downloaded_path}}")
else:
    print(f"[MODEL] Found cached model at: {{model_path}}")

# Step 4: Launch llama-server with Dual-GPU offload
print("\\n[STEP 4] Launching llama-server in background with GPU offloading...")
server_cmd = [
    LLAMA_SERVER,
    "-m", model_path,
    ''' + server_args_str + '''
]

print("[SERVER CMD]:", " ".join(server_cmd))
server_log = open("/tmp/llama_server.log", "w")
server_proc = subprocess.Popen(server_cmd, stdout=server_log, stderr=subprocess.STDOUT)
print(f"[SERVER] Process spawned with PID: {server_proc.pid}")

# Poll health endpoint
print("[SERVER] Waiting for model to load into VRAM...")
ready = False
for attempt in range(120):
    time.sleep(3)
    try:
        req = urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
        if req.status == 200:
            print(f"\\n[SERVER READY] Model is fully loaded and accepting requests on port 8080! (attempt {attempt+1})")
            ready = True
            break
    except Exception:
        sys.stdout.write(".")
        sys.stdout.flush()

if not ready:
    print("\\n[ERROR] Server failed to become ready within timeout. Tail log:")
    server_log.flush()
    with open("/tmp/llama_server.log") as f:
        print(f.read()[-3000:])
    sys.exit(1)

# Step 5: Start Tunnel (Cloudflare Quick Tunnel)
print("\\n[STEP 5] Setting up secure outbound tunnel...")
tunnel_url = ""

cf_bin = "/tmp/bin/cloudflared"
if not os.path.exists(cf_bin):
    print("[TUNNEL] Downloading Cloudflare tunnel binary...")
    subprocess.run(["curl", "-sL", "-o", cf_bin, "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"], check=True)
    os.chmod(cf_bin, 0o755)

cf_log_path = "/tmp/cloudflared.log"
cf_log = open(cf_log_path, "w")
print("[TUNNEL] Starting Cloudflare Quick Tunnel to http://127.0.0.1:8080...")
cf_proc = subprocess.Popen([cf_bin, "tunnel", "--url", "http://127.0.0.1:8080", "--no-autoupdate"], stdout=cf_log, stderr=subprocess.STDOUT)

# Extract TryCloudflare URL
for _ in range(60):
    time.sleep(1)
    if os.path.exists(cf_log_path):
        with open(cf_log_path, "r", errors="ignore") as f:
            content = f.read()
            match = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", content)
            if match:
                tunnel_url = match.group(0)
                print(f"\\n[TUNNEL ESTABLISHED] URL: {tunnel_url}")
                break

if not tunnel_url:
    print("[ERROR] Could not extract tunnel URL from Cloudflare logs. Log contents:")
    with open(cf_log_path) as f:
        print(f.read())
    sys.exit(1)

# Step 6: Notify VPS Webhook
print("\\n[STEP 6] Reporting tunnel URL to VPS Manager...")
try:
    payload = json.dumps({"tunnel_url": tunnel_url, "model_id": MODEL_ID, "status": "ready"}).encode("utf-8")
    req = urllib.request.Request(
        WEBHOOK_TARGET,
        data=payload,
        headers={"Content-Type": "application/json", "User-Agent": "KaggleKernel/1.0"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(f"[WEBHOOK] Successfully notified VPS manager (HTTP {resp.status}): {resp.read().decode()}")
except Exception as e:
    print(f"[WEBHOOK] Note: Could not auto-post to webhook at {WEBHOOK_TARGET} ({e}).")
    print(f"Tunnel URL is printed below for manual connection if needed.")

print("\\n" + "=" * 70)
print(f"=== TUNNEL_URL: {tunnel_url} ===")
print("=" * 70 + "\\n")

# Step 7: Continuous Keepalive Loop
print("[STEP 7] Entering keepalive loop. Serving inference requests...")
tick = 0
while True:
    time.sleep(30)
    tick += 1
    # Check llama-server process health
    if server_proc.poll() is not None:
        print("[CRITICAL] llama-server process exited unexpectedly!")
        with open("/tmp/llama_server.log") as f:
            print(f.read()[-2000:])
        break
        
    # Check tunnel health
    if cf_proc.poll() is not None:
        print("[CRITICAL] cloudflared process exited unexpectedly!")
        break
        
    if tick % 4 == 0:
        print(f"[HEARTBEAT] Elapsed: {tick * 30}s | Upstream {tunnel_url} is healthy and serving.")
'''
        return script_code

    def prepare_bundle(self, model_id: str) -> Path:
        BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
        script_content = self.generate_kernel_script(model_id)
        kernel_py = BUNDLE_DIR / "kernel.py"
        kernel_py.write_text(script_content, encoding="utf-8")

        username = self.load_kaggle_credentials().get("username", "samirandas22")
        slug_name = "hermes-uncensored-engine"
        
        metadata = {
            "id": f"{username}/{slug_name}",
            "title": "Hermes Uncensored Engine",
            "code_file": "kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }
        
        meta_json = BUNDLE_DIR / "kernel-metadata.json"
        meta_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.log(f"Prepared kernel bundle in {BUNDLE_DIR} for model '{model_id}'")
        return BUNDLE_DIR

    def push_kernel(self, model_id: str) -> Dict[str, Any]:
        self.active_model_id = model_id
        self.prepare_bundle(model_id)
        self.set_status("initializing", f"Pushing kernel for {model_id} to Kaggle...")
        self.launch_time = time.time()

        kaggle_bin = str(KAGGLE_VENV_BIN) if KAGGLE_VENV_BIN.exists() else "kaggle"
        cmd = [kaggle_bin, "kernels", "push", "-p", str(BUNDLE_DIR)]

        try:
            self.log(f"Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd, env=self.get_subprocess_env(), capture_output=True, text=True, timeout=60)
            output = res.stdout + "\n" + res.stderr
            self.log(f"Kaggle push output:\n{output.strip()}")

            if res.returncode == 0:
                self.set_status("downloading", "Kernel successfully pushed and queued on Kaggle GPU!")
                self.start_background_polling()
                return {"success": True, "output": output, "status": self.status}
            else:
                self.set_status("error", f"Kaggle push failed (exit code {res.returncode}): {output.strip()}")
                return {"success": False, "error": output, "status": self.status}
        except Exception as e:
            self.set_status("error", f"Exception pushing to Kaggle: {e}")
            return {"success": False, "error": str(e), "status": self.status}

    def start_background_polling(self):
        self._stop_polling.set()
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=2)
        self._stop_polling.clear()
        self._polling_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._polling_thread.start()

    def _poll_loop(self):
        self.log("Started background monitor for Kaggle kernel...")
        kaggle_bin = str(KAGGLE_VENV_BIN) if KAGGLE_VENV_BIN.exists() else "kaggle"
        username = self.load_kaggle_credentials().get("username", "samirandas22")
        kernel_ref = f"{username}/hermes-uncensored-engine"
        out_dir = Path("/root/uncensored-manager/kernel_output")
        out_dir.mkdir(parents=True, exist_ok=True)

        attempts = 0
        while not self._stop_polling.is_set() and attempts < 120:
            attempts += 1
            time.sleep(10)
            
            # 1. Check kernel status
            try:
                cmd = [kaggle_bin, "kernels", "status", kernel_ref]
                res = subprocess.run(cmd, env=self.get_subprocess_env(), capture_output=True, text=True, timeout=30)
                status_text = res.stdout.strip()
                self.log(f"[Kaggle Status]: {status_text}")

                if "running" in status_text.lower():
                    if self.status in ["initializing", "downloading"]:
                        self.set_status("loading_vram", "Kernel is running on Kaggle! Loading model & tunnel...")
                elif any(w in status_text.lower() for w in ["cancel", "canceled", "cancelled"]):
                    self.set_status("idle", f"Kernel run cancelled: {status_text}")
                    self.tunnel_url = ""
                    break
                elif "error" in status_text.lower():
                    self.set_status("error", f"Kernel error on Kaggle: {status_text}")
                    break
            except Exception as e:
                logger.warning(f"Error checking kernel status: {e}")

            # 2. If tunnel_url is not set yet, check kernel output
            if not self.tunnel_url:
                try:
                    # Use Kaggle API to get real-time stream logs without downloading giant output files
                    try:
                        import urllib.request, json as _json, base64 as _base64
                        creds = self.load_kaggle_credentials()
                        if creds.get("username") and creds.get("key_present"):
                            with open(KAGGLE_CONFIG_PATH) as kf:
                                kdata = _json.load(kf)
                            auth_str = _base64.b64encode(f"{kdata['username']}:{kdata['key']}".encode()).decode()
                            api_url = f"https://www.kaggle.com/api/v1/kernels/output?userName={kdata['username']}&kernelSlug=hermes-uncensored-engine"
                            req = urllib.request.Request(api_url, headers={"Authorization": f"Basic {auth_str}"})
                            with urllib.request.urlopen(req, timeout=10) as rresp:
                                rdata = _json.loads(rresp.read().decode())
                                rlog = rdata.get("log", "")
                                rtext = ""
                                if isinstance(rlog, str):
                                    try:
                                        for rentry in _json.loads(rlog):
                                            rtext += rentry.get("data", "")
                                    except Exception:
                                        rtext = rlog
                                match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", rtext)
                                if match:
                                    found_url = match.group(0)
                                    self.log(f"[AUTODISCOVERY] Found active Cloudflare tunnel: {found_url}")
                                    self.set_tunnel_url(found_url)
                                    self.set_status("ready", f"Live Tunnel: {found_url}")
                    except Exception as e_poll:
                        logger.debug(f"Fast API poll exception: {e_poll}")
                except Exception as e:
                    logger.debug(f"Output poll check: {e}")

            if self.tunnel_url and self.status == "ready":
                # Verified live, continue light heartbeat check
                time.sleep(20)

    def get_elapsed_seconds(self) -> int:
        if self.launch_time:
            return int(time.time() - self.launch_time)
        return 0

    def stop_kernel(self) -> Dict[str, Any]:
        """
        Stops the active Kaggle kernel to conserve GPU quota,
        terminates background polling, tears down tunnel, and resets status to idle.
        """
        self.log("Received Stop Engine request. Initiating shutdown...")
        # 1. Stop background polling thread immediately
        self._stop_polling.set()
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=2)

        # 2. Reset active tunnel and elapsed time
        self.tunnel_url = ""
        self.launch_time = None
        self.set_status("idle", "Engine stopped by operator. GPU quota conserved.")

        # 3. Stop/cancel kernel on Kaggle
        kaggle_bin = str(KAGGLE_VENV_BIN) if KAGGLE_VENV_BIN.exists() else "kaggle"
        username = self.load_kaggle_credentials().get("username", "samirandas22")
        kernel_ref = f"{username}/hermes-uncensored-engine"

        stopped_via_cli = False
        for cmd_verb in ["stop", "cancel"]:
            try:
                res = subprocess.run(
                    [kaggle_bin, "kernels", cmd_verb, kernel_ref],
                    env=self.get_subprocess_env(),
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if res.returncode == 0:
                    self.log(f"Kaggle kernel stopped via CLI '{cmd_verb}': {res.stdout.strip()}")
                    stopped_via_cli = True
                    break
            except Exception:
                pass

        if not stopped_via_cli:
            try:
                self.log("Pushing cancellation script to release Kaggle GPU resource...")
                stop_dir = Path("/root/uncensored-manager/stop_bundle")
                stop_dir.mkdir(parents=True, exist_ok=True)

                stop_script = (
                    "#!/usr/bin/env python3\\n"
                    "# Autonomous Uncensored Engine - Shutdown Sentinel\\n"
                    "import sys\\n"
                    "print('[SHUTDOWN] Engine stop requested by operator. Releasing GPU.')\\n"
                    "sys.exit(0)\\n"
                )
                (stop_dir / "kernel.py").write_text(stop_script, encoding="utf-8")

                metadata = {
                    "id": kernel_ref,
                    "title": "Hermes Uncensored Engine",
                    "code_file": "kernel.py",
                    "language": "python",
                    "kernel_type": "script",
                    "is_private": True,
                    "enable_gpu": False,
                    "enable_tpu": False,
                    "enable_internet": False,
                    "dataset_sources": [],
                    "competition_sources": [],
                    "kernel_sources": [],
                    "model_sources": []
                }
                (stop_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                cmd = [kaggle_bin, "kernels", "push", "-p", str(stop_dir)]
                res = subprocess.run(cmd, env=self.get_subprocess_env(), capture_output=True, text=True, timeout=30)
                if res.returncode == 0:
                    self.log("Kaggle run preempted & cancelled. Active GPU session terminated.")
                else:
                    self.log(f"Notice: Push stop script returned code {res.returncode}: {res.stderr.strip()}")
            except Exception as e:
                self.log(f"Stop kernel push exception: {e}")

        if self.config_callback:
            self.config_callback()

        return {"success": True, "status": self.status, "message": "Engine stopped and tunnel torn down."}

    def get_state(self) -> Dict[str, Any]:
        effective_status = "ready" if self.tunnel_url else self.status
        effective_detail = f"Live Tunnel: {self.tunnel_url}" if self.tunnel_url else self.status_detail
        return {
            "status": effective_status,
            "status_detail": effective_detail,
            "active_model_id": self.active_model_id,
            "active_model": MODELS.get(self.active_model_id, MODELS[DEFAULT_MODEL_ID]),
            "tunnel_url": self.tunnel_url,
            "tunnel_type": self.tunnel_type,
            "elapsed_seconds": self.get_elapsed_seconds(),
            "kernel_slug": self.kernel_slug,
            "recent_logs": self.recent_logs[-50:],
            "available_models": list(MODELS.values())
        }
