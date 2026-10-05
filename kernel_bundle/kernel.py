#!/usr/bin/env python3
# ==============================================================================
# Autonomous Uncensored AI Agent Kernel - Dual Tesla T4 / P100 Execution Engine
# Model: CyberTiel-Coder-35B-A3B (Abliterated MoE) (35B MoE (~3.4B active), UD-Q4_K_XL (Cyber-imatrix))
# Context Window: 262,144 tokens (262k / ~2.75 lakh) | RoPE: YaRN scale 8 (orig 32k) | KV: q4_0 / q4_0
# Hugging Face: peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF / Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf
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

MODEL_NAME = "CyberTiel-Coder-35B-A3B (Abliterated MoE)"
MODEL_REPO = "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF"
MODEL_FILENAME = "Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf"
MODEL_FILESIZE = 22.36
MODEL_ID = "cybertiel-coder-35b-a3b"
WEBHOOK_TARGET = "http://164.68.114.240:8778/api/tunnel/register"

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
print("\n[STEP 1] Installing huggingface_hub and requests...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "requests"], check=False)

# Step 2: Acquire llama.cpp binary (Prebuilt CUDA or fast source build)
LLAMA_SERVER = "/tmp/bin/llama-server"

def setup_llama_server():
    global LLAMA_SERVER
    print("\n[STEP 2] Setting up llama-server with GPU support...")
    
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
                print(f"[LLAMA] Successfully installed pre-built llama-server at: {LLAMA_SERVER}")
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
            print(f"[LLAMA] Compiled llama-server at: {LLAMA_SERVER}")
            return True
            
    raise RuntimeError("Could not establish llama-server binary!")

setup_llama_server()

# Step 3: Download Model GGUF
print(f"\n[STEP 3] Downloading GGUF model: {MODEL_REPO} / {MODEL_FILENAME}...")
from huggingface_hub import hf_hub_download

model_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
if not os.path.exists(model_path):
    print(f"[MODEL] Fetching {MODEL_FILENAME} (~{MODEL_FILESIZE} GB)...")
    downloaded_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=MODEL_DIR
    )
    print(f"[MODEL] Model downloaded successfully to: {downloaded_path}")
else:
    print(f"[MODEL] Found cached model at: {model_path}")

# Step 4: Launch llama-server with Dual-GPU offload
print("\n[STEP 4] Launching llama-server in background with GPU offloading...")
server_cmd = [
    LLAMA_SERVER,
    "-m", model_path,
    "-np", "1",
    "--rope-scaling", "yarn", "--rope-scale", "8", "--yarn-orig-ctx", "32768", "-c", "262144", "-ctk", "q4_0", "-ctv", "q4_0", "-ngl", "52", "-sm", "layer", "-ts", "1,1", "-b", "2048", "-ub", "512", "-fa", "auto", "--threads", "4"
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
            print(f"\n[SERVER READY] Model is fully loaded and accepting requests on port 8080! (attempt {attempt+1})")
            ready = True
            break
    except Exception:
        sys.stdout.write(".")
        sys.stdout.flush()

if not ready:
    print("\n[ERROR] Server failed to become ready within timeout. Tail log:")
    server_log.flush()
    with open("/tmp/llama_server.log") as f:
        print(f.read()[-3000:])
    sys.exit(1)

# Step 5: Start Tunnel (Cloudflare Quick Tunnel)
print("\n[STEP 5] Setting up secure outbound tunnel...")
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
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
            if match:
                tunnel_url = match.group(0)
                print(f"\n[TUNNEL ESTABLISHED] URL: {tunnel_url}")
                break

if not tunnel_url:
    print("[ERROR] Could not extract tunnel URL from Cloudflare logs. Log contents:")
    with open(cf_log_path) as f:
        print(f.read())
    sys.exit(1)

# Step 6: Notify VPS Webhook
print("\n[STEP 6] Reporting tunnel URL to VPS Manager...")
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

print("\n" + "=" * 70)
print(f"=== TUNNEL_URL: {tunnel_url} ===")
print("=" * 70 + "\n")

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
