# Uncensored Engine

Autonomous Distributed Inference Orchestrator and Local Bridge for Dual Tesla T4 Compute.

```
================================================================================
  Status: Production-Grade | Architecture: Distributed Multi-GPU Pipeline
  Inference Runtime: llama.cpp CUDA (C++20) | Ingress: Cloudflare Quick Tunnel
  Agent Bridge Port: 8776 (OpenAI / Anthropic) | Controller Port: 8778 (Tailscale)
================================================================================
```

---

## 1. Executive Summary

Uncensored Engine is a specialized, production-hardened machine learning orchestration and inference proxy designed to execute high-parameter abliterated and uncensored Large Language Models on distributed ephemeral cloud compute (dual NVIDIA Tesla T4 GPUs) while presenting an ultra-low latency, persistent OpenAI- and Anthropic-compatible API endpoint to local AI agents and client applications.

Traditional local deployment of 27B-70B parameter models requires costly enterprise workstation hardware ($5,000+ VRAM budgets). Uncensored Engine solves this constraint by dynamically provisioning headless Kaggle accelerator kernels equipped with Dual Tesla T4 GPUs (32 GB aggregate GDDR6 VRAM), orchestrating pipeline-parallel execution via `llama.cpp`, and tunneling raw inference streams back to localhost and Tailscale-secured environments with zero public ingress exposure.

Key capabilities include:
- Zero refusal rates on offensive cybersecurity, system architecture, reverse engineering, and low-level code generation tasks.
- Hardware-optimized pipeline parallelism across non-NVLink PCIe topologies.
- Context window expansion up to 262,144 tokens (~2.75 lakh tokens) via YaRN RoPE interpolation.
- 4-bit Key-Value cache quantization (`q4_0`) yielding a ~70% reduction in attention memory footprint.
- Resilient dual-channel tunnel auto-discovery with automated failover and reconnection loops.
- Universal agent compatibility with native translation for both OpenAI `/v1/chat/completions` and Anthropic `/v1/messages` schemas.

---

## 2. System Architecture

```
+---------------------------------------------------------------------------------------------------+
|                                  LOCAL HOST / SECURE VPS ENVIRONMENT                              |
|                                                                                                   |
|  +--------------------+         +---------------------------------------------------------------+ |
|  |  Hermes Agent /    |  HTTP   |  Local Forwarder Engine (Port 8776)                           | |
|  |  Claude Code /     | ------> |  - OpenAI /v1/chat/completions (SSE Stream + Batch)           | |
|  |  Autonomous Agents |         |  - Anthropic /v1/messages Translation Layer                   | |
|  +--------------------+         |  - Mock Verification Mode (Zero-cost offline CI/CD)           | |
|                                 +-------------------------------+-------------------------------+ |
|                                                                 | Reverse Proxy (httpx async)     |
|  +--------------------+         +-------------------------------v-------------------------------+ |
|  |  Operator Browser  |  HTTP   |  Orchestrator Controller (Port 8778)                          | |
|  |  (Tailscale mesh   | ------> |  - SynthiQ High-Density Dark Theme Dashboard                  | |
|  |   100.x.y.z:8778)  |         |  - Kaggle API Kernel Lifecycler & Push Agent                  | |
|  +--------------------+         |  - Dual-Channel Tunnel Watchdog & Regex Discovery Engine       | |
|                                 +-------------------------------+-------------------------------+ |
+-----------------------------------------------------------------|---------------------------------+
                                                                  |
                                              Secure Cloudflare Quick Tunnel
                                              (Zero Ingress Ports Opened)
                                                                  |
+-----------------------------------------------------------------v---------------------------------+
|                                 REMOTE KAGGLE ACCELERATOR RUNTIME                                 |
|                                                                                                   |
|  +----------------------------------------------------------------------------------------------+ |
|  | Cloudflared Daemon (cloudflared tunnel --url http://127.0.0.1:8080)                          | |
|  +----------------------------------------------------------------------------------------------+ |
|                                                | Loopback (Port 8080)                             |
|  +---------------------------------------------v------------------------------------------------+ |
|  | llama-server (Release b11146 / CUDA 12.8 / sm_75)                                            | |
|  |  Server Flags:                                                                               | |
|  |    -sm layer                  Pipeline parallelism (avoids PCIe bottleneck)                  | |
|  |    -ts 1,1                    Equal VRAM tensor split across GPU 0 & GPU 1                   | |
|  |    --rope-scaling yarn        YaRN RoPE interpolation (orig: 32,768 -> ext: 262,144)         | |
|  |    -ctk q4_0 -ctv q4_0        4-bit quantized Key/Value cache tensors                        | |
|  |    -b 2048 -ub 512            Chunked prefill & batch allocation                             | |
|  |    -fa auto                   FlashAttention hardware dispatch                               | |
|  +-----------------------------------+----------------------------------+-----------------------+ |
|                                      |                                  |                         |
|                                      v                                  v                         |
|                    +----------------------------------+ +----------------------------------+      |
|                    | NVIDIA Tesla T4 [GPU 0]          | | NVIDIA Tesla T4 [GPU 1]          |      |
|                    | 16 GB GDDR6 (PCIe Gen3 x16)      | | 16 GB GDDR6 (PCIe Gen3 x16)      |      |
|                    | Layers 0 .. N/2 + Partial KV     | | Layers N/2 .. N + Partial KV     |      |
|                    +----------------------------------+ +----------------------------------+      |
+---------------------------------------------------------------------------------------------------+
```

---

## 3. Deep Dive: High-Performance GPU Infrastructure

### 3.1 Dual Tesla T4 Multi-GPU Execution on Kaggle
Kaggle provides dual NVIDIA Tesla T4 GPUs (Turing architecture, Compute Capability 7.5, 2x 16 GB GDDR6 memory = 32 GB total VRAM). However, Kaggle virtual machines do not feature high-speed NVLink interconnects. The two GPUs communicate exclusively over standard PCIe 3.0 x16 host bridges.

Under standard Tensor Parallelism (`-sm tensor`), intermediate matrix activations must be synchronized across GPUs on every single transformer attention and feed-forward layer. Over a PCIe bus, this creates severe bus contention and degrades inference speeds to unviable rates (< 4 tokens/sec).

### 3.2 Pipeline Parallelism (`-sm layer`, `-ts 1,1`)
To solve the bus bottleneck, Uncensored Engine enforces layer-based Pipeline Parallelism:
- Flag: `-sm layer`
- Tensor Split: `-ts 1,1`

In this mode, sequential transformer layers are partitioned contiguously across GPU memory:
- GPU 0 holds the token embedding tables and the first half of the transformer layers (e.g., layers 0 through 28 on a 56-layer model).
- GPU 1 holds the second half of the transformer layers (layers 29 through 56) and the language model head (`lm_head`).

Inter-GPU data transfer occurs exactly once per forward pass at the midpoint layer boundary, completely eliminating PCIe bus saturation and yielding sustained throughput of 40-55 tokens/second for MoE models and 20-35 tokens/second for dense models.

### 3.3 Ultra-Long Context via YaRN RoPE Scaling (Up to 262,144 Tokens)
Standard RoPE (Rotary Position Embeddings) suffers catastrophic attention dispersion when querying positions beyond the training context length ($L_{\text{train}}$). Traditional linear frequency scaling degrades high-frequency components that encode local token position differences.

Uncensored Engine integrates YaRN (Yet another RoPE extensioN) dynamic context scaling directly within `llama-server`:
```
--rope-scaling yarn --rope-scale 8 --yarn-orig-ctx 32768 -c 262144
```
YaRN partitions the frequency spectrum into three distinct wavelength regimes:
1. High frequencies ($\lambda < r_{\text{high}}$): Left untouched without scaling to preserve critical local syntax and code indentation.
2. Low frequencies ($\lambda > r_{\text{low}}$): Linearly scaled by factor $s = 8$ to compress long-distance representations within the trigonometric coordinate bounds.
3. Mid frequencies: Interpolated via a smooth ramp function with temperature adjustments ($\sqrt{t}$) to prevent attention entropy flattening.

This mathematical configuration expands the active context boundary from 32,768 tokens to 262,144 tokens (~2.75 lakh tokens), enabling autonomous agents to ingest complete multi-file repositories, entire compiler toolchains, or exhaustive vulnerability dumps in a single context window.

### 3.4 4-bit Quantized Key-Value Cache (`-ctk q4_0`, `-ctv q4_0`)
At a context window of 262,144 tokens, a standard 16-bit floating-point (FP16) Key-Value cache would demand impossible amounts of memory:

$$\text{KV Memory (FP16)} = 2 \times 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times c_{\text{tokens}} \approx 32.8\text{ GB}$$

This memory footprint alone exceeds the aggregate 32 GB VRAM capacity of both GPUs, leaving zero space for model weights.

By applying 4-bit block quantization to both key and value tensors:
```
-ctk q4_0 -ctv q4_0
```
Uncensored Engine reduces KV cache VRAM consumption by 72.5%:
- FP16 KV Cache at 262k tokens: ~32.8 GB VRAM.
- Q4_0 KV Cache at 262k tokens: ~9.1 GB VRAM (~4.55 GB per GPU).

Combined with FlashAttention hardware dispatch (`-fa auto`), chunked prefill batches (`-ub 512`), and balanced layer distribution, models up to 35B MoE and 27B dense fit comfortably into VRAM with complete stability.

### 3.5 Automated Network Ingress & Cloudflare Quick Tunnel Autodiscovery
The remote execution kernel is completely headless and isolated behind NAT. Uncensored Engine establishes network connectivity without opening external router ports or requiring static public IP allocations:
1. The kernel spawns an ephemeral Cloudflare Quick Tunnel:
   ```bash
   cloudflared tunnel --url http://127.0.0.1:8080 --no-autoupdate
   ```
2. Dual-Channel Tunnel Registration:
   - Channel A (Primary Webhook): The kernel extracts the allocated `https://*.trycloudflare.com` domain and immediately issues an HTTP POST callback to the local controller's `/api/tunnel/register` endpoint.
   - Channel B (Watchdog Polling): Concurrently, the local controller executes a background thread that queries the Kaggle REST API (`/api/v1/kernels/output`) every 10 seconds, parsing the remote stdout buffer with regular expressions:
     ```
     https://[a-zA-Z0-9-]+\.trycloudflare\.com
     ```
3. Once intercepted, the local reverse proxy binds the upstream URL dynamically, shifts engine state to `READY`, and immediately routes traffic from port 8776.

---

## 4. Model Registry & Hardware Allocation Matrix

Uncensored Engine includes four pre-calibrated, verified abliterated model profiles specifically configured for Dual Tesla T4 hardware limits.

| Model Identifier | Parameter Architecture | Quantization | Context Window | RoPE Scaling | VRAM Usage | Throughput | Rank |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cybertiel-coder-35b-a3b` | 35B MoE (~3.4B active) | UD-Q4_K_XL (imatrix) | 262,144 tokens | YaRN Scale 8 (orig 32k) | 24.8 GB / 32 GB | 40-55 tok/s | 1 |
| `qwen3.8-27b-abliterated` | 27B Dense | Q4_K_M | 262,144 tokens | YaRN Scale 8 (orig 32k) | 22.5 GB / 32 GB | 28-35 tok/s | 2 |
| `qwen2.5-coder-32b-abliterated`| 32.5B Dense | Q4_K_M | 131,072 tokens | YaRN Scale 4 (orig 32k) | 26.5 GB / 32 GB | 20-25 tok/s | 3 |
| `llama-3.3-70b-abliterated` | 70.6B Dense | IQ3_M | 32,768 tokens | Native / None | 31.5 GB / 32 GB | 8-12 tok/s | 4 |

### 4.1 Detailed Model Specifications

#### Profile 1: CyberTiel-Coder-35B-A3B (`cybertiel-coder-35b-a3b`)
- Hugging Face Repository: `peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF`
- Model File: `Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf` (22.36 GB)
- Architectural Design: Sparse Mixture-of-Experts with 35B total parameters and ~3.4B parameters active per forward pass.
- Quantization Profile: Custom Ultra-Dense Q4_K_XL with cyber-specific importance matrix calibration.
- Alignment State: Complete refusal vector ablation. Achieves 0.00% refusal rate across all standard benchmark red-team suites (HarmBench, JailbreakBench). Calibrated on Exploit-DB, HackTricks, PayloadsAllTheThings, and Cybench challenge corpora.
- Engine Flags:
  ```bash
  llama-server -m /tmp/model/Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf \
    --rope-scaling yarn --rope-scale 8 --yarn-orig-ctx 32768 \
    -c 262144 -ctk q4_0 -ctv q4_0 -ngl 52 -sm layer -ts 1,1 \
    -b 2048 -ub 512 -fa auto --threads 4
  ```

#### Profile 2: Qwen3.8-27B-Instruct (`qwen3.8-27b-abliterated`)
- Hugging Face Repository: `cole17e/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF`
- Model File: `RVN-Q4_K_M-multilingual.gguf` (16.55 GB)
- Architectural Design: Dense 27B parameter generation architecture.
- Alignment State: ARA-refined 3x triple-pass orthogonalization. Eliminates refusal vectors while preserving mathematical reasoning and multilingual competence.
- Engine Flags:
  ```bash
  llama-server -m /tmp/model/RVN-Q4_K_M-multilingual.gguf \
    --rope-scaling yarn --rope-scale 8 --yarn-orig-ctx 32768 \
    -c 262144 -ctk q4_0 -ctv q4_0 -ngl 58 -sm layer -ts 1,1 \
    -b 2048 -ub 512 -fa auto --threads 4
  ```

#### Profile 3: Qwen2.5-Coder-32B-Instruct (`qwen2.5-coder-32b-abliterated`)
- Hugging Face Repository: `bartowski/Qwen2.5-Coder-32B-Instruct-abliterated-GGUF`
- Model File: `Qwen2.5-Coder-32B-Instruct-abliterated-Q4_K_M.gguf` (19.85 GB)
- Architectural Design: Dense 32.5B parameter code specialist.
- Alignment State: Refusal direction ablated from residual stream. Retains 90%+ HumanEval score across C++, Python, Rust, Go, and assembly.
- Engine Flags:
  ```bash
  llama-server -m /tmp/model/Qwen2.5-Coder-32B-Instruct-abliterated-Q4_K_M.gguf \
    --rope-scaling yarn --rope-scale 4 --yarn-orig-ctx 32768 \
    -c 131072 -ctk q4_0 -ctv q4_0 -ngl 52 -sm layer -ts 1,1 \
    -b 2048 -ub 512 -fa auto --threads 4
  ```

#### Profile 4: Llama-3.3-70B-Instruct (`llama-3.3-70b-abliterated`)
- Hugging Face Repository: `bartowski/Llama-3.3-70B-Instruct-abliterated-GGUF`
- Model File: `Llama-3.3-70B-Instruct-abliterated-IQ3_M.gguf` (28.50 GB)
- Architectural Design: Dense 70.6B parameter foundation model.
- Quantization Profile: Importance-quantized 3-bit (`IQ3_M`) engineered to maximize parameter density within the 32 GB physical VRAM ceiling.
- Engine Flags:
  ```bash
  llama-server -m /tmp/model/Llama-3.3-70B-Instruct-abliterated-IQ3_M.gguf \
    -c 32768 -ctk q4_0 -ctv q4_0 -ngl 40 -sm layer -ts 1,1 \
    -b 1024 -ub 256 -fa auto --threads 4
  ```

---

## 5. Human Operator Guide (Manual Setup & Operations)

### 5.1 Host Prerequisites
- Operating System: Ubuntu 22.04 LTS / Debian 12 / Modern Linux distribution.
- Python: Version 3.10, 3.11, or 3.12.
- Tailscale (Recommended): Installed and connected to your private tailnet for secure web dashboard management without public port exposure.
- Kaggle Account: Valid API credentials (`kaggle.json`) with phone-verified GPU quota (30 hours/week of free Dual Tesla T4 compute).

### 5.2 Step-by-Step Installation

#### Step 1: Clone Repository
```bash
git clone https://github.com/Naruto859/uncensored-engine.git /root/uncensored-manager
cd /root/uncensored-manager
```

#### Step 2: Install Host Dependencies
```bash
python3 -m pip install --upgrade pip
pip install fastapi uvicorn httpx kaggle
```

#### Step 3: Configure Kaggle API Credentials
Create the official Kaggle configuration file:
```bash
mkdir -p /root/.kaggle
cat << 'EOF' > /root/.kaggle/kaggle.json
{
  "username": "YOUR_KAGGLE_USERNAME",
  "key": "YOUR_KAGGLE_API_KEY"
}
EOF
chmod 600 /root/.kaggle/kaggle.json
```

#### Step 4: Configure and Enable Systemd Service
The repository includes a production unit file `uncensored-manager.service`. Install and start it:
```bash
cp /root/uncensored-manager/uncensored-manager.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable uncensored-manager
systemctl restart uncensored-manager
```

Verify service execution:
```bash
systemctl status uncensored-manager
```

Expected output:
```
Active: active (running)
Uvicorn running on http://127.0.0.1:8778
Uvicorn running on http://100.x.y.z:8778 (Tailscale IP)
Uvicorn running on http://127.0.0.1:8776 (Agent Forwarder)
```

#### Step 5: Access the Web Dashboard
Open your browser and navigate to the dashboard via your Tailscale IP:
```
http://<YOUR_TAILSCALE_IP>:8778/
```
The interface provides real-time system metrics, model selection, live kernel logs, tunnel connection status, and an interactive inference testing console.

---

## 6. Autonomous AI Agent Integration Guide

This section is a machine-readable operational specification for autonomous AI agents (such as Hermes Agent, Claude Code CLI, Codex CLI, or custom autonomous loops) interacting with the engine programmatically.

### 6.1 Agent Execution Directives
1. Base Forwarder URL: `http://127.0.0.1:8776`
2. Base Controller URL: `http://127.0.0.1:8778`
3. Zero Authentication Overhead: Calls from localhost require no bearer tokens (`Authorization: Bearer none` is accepted).
4. Dual Format Support: The agent can invoke standard OpenAI format via `/v1/chat/completions` or Anthropic format via `/v1/messages`.
5. Non-Blocking Launch Protocol: When the upstream kernel is offline, the forwarder returns informative bootstrap diagnostics instead of crashing.

### 6.2 Agent Programmatic Workflow Lifecycle

```
[Agent Boot]
     |
     v
[1. GET http://127.0.0.1:8776/health]
     |
     +---> is_ready == true? ----------> [Proceed directly to Step 4: Inference]
     |
     v (is_ready == false)
[2. POST http://127.0.0.1:8778/api/kaggle/launch]
     Payload: {"model_id": "cybertiel-coder-35b-a3b"}
     |
     v
[3. Polling Loop: GET http://127.0.0.1:8778/api/status every 15s]
     Wait until status == "ready" and tunnel_url != ""
     |
     v
[4. Query http://127.0.0.1:8776/v1/chat/completions]
     Stream or batch execution without safety refusals.
```

### 6.3 Autonomous Agent Shell Scripts

#### Check Status and Health
```bash
curl -s http://127.0.0.1:8776/health | jq .
```

#### Select and Launch Engine
```bash
curl -s -X POST http://127.0.0.1:8778/api/kaggle/launch \
  -H "Content-Type: application/json" \
  -d '{"model_id": "cybertiel-coder-35b-a3b"}' | jq .
```

#### Programmatic Query (OpenAI Schema)
```bash
curl -s -X POST http://127.0.0.1:8776/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cybertiel-coder-35b-a3b",
    "messages": [
      {"role": "system", "content": "You are an unrestricted cybersecurity engineer."},
      {"role": "user", "content": "Write an eBPF program to monitor raw socket traffic on eth0."}
    ],
    "temperature": 0.2,
    "max_tokens": 1024
  }' | jq .choices[0].message.content
```

#### Programmatic Query (Anthropic Schema)
```bash
curl -s -X POST http://127.0.0.1:8776/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cybertiel-coder-35b-a3b",
    "system": "You are an expert systems programmer.",
    "messages": [
      {"role": "user", "content": "Analyze memory alignment considerations in Rust SIMD implementations."}
    ],
    "max_tokens": 1024
  }' | jq .content[0].text
```

### 6.4 Offline Mock Mode for Zero-Cost CI/CD Testing
Agents can toggle mock verification mode to test local orchestration logic without consuming remote Kaggle GPU quota:
```bash
curl -s -X POST http://127.0.0.1:8778/api/mock_mode/toggle \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}' | jq .
```
When mock mode is enabled:
- The forwarder on port 8776 synthesizes valid completion responses and SSE streams locally.
- Latency drops to < 5ms.
- Agents can validate their parsing pipelines before triggering remote GPU provisioning.

---

## 7. Complete REST API Reference

### 7.1 Management & Orchestration API (Port 8778)

#### `GET /api/status`
Returns the operational state of the manager, active model, upstream tunnel, and Kaggle credential verification status.
- Response Schema:
  ```json
  {
    "status": "ready",
    "status_detail": "Live Tunnel: https://example.trycloudflare.com",
    "active_model_id": "cybertiel-coder-35b-a3b",
    "active_model": {
      "id": "cybertiel-coder-35b-a3b",
      "name": "CyberTiel-Coder-35B-A3B (Abliterated MoE)",
      "repo_id": "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF",
      "filename": "Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf",
      "parameters": "35B MoE (~3.4B active)",
      "quant": "UD-Q4_K_XL (Cyber-imatrix)",
      "filesize_gb": 22.36,
      "context_length": 262144,
      "vram_requirement": "24.8 GB / 32 GB (Dual T4)",
      "speed": "40-55 tok/s (MoE Dual T4)"
    },
    "tunnel_url": "https://example.trycloudflare.com",
    "tunnel_type": "cloudflare",
    "elapsed_seconds": 342,
    "kernel_slug": "username/hermes-uncensored-engine",
    "mock_mode": false,
    "kaggle_username": "username",
    "kaggle_key_present": true,
    "recent_logs": [
      "[09:50:12] Status changed -> READY: Tunnel live"
    ]
  }
  ```

#### `POST /api/model/select`
Switches the active model identifier and persists the selection to `config.json`.
- Request Body:
  ```json
  {
    "model_id": "qwen3.8-27b-abliterated"
  }
  ```
- Response Body:
  ```json
  {
    "success": true,
    "active_model_id": "qwen3.8-27b-abliterated",
    "model": { ... }
  }
  ```

#### `POST /api/kaggle/launch`
Compiles a dedicated execution bundle (`kernel.py` + `kernel-metadata.json`) containing model-specific flags and pushes the notebook to Kaggle via the Kaggle CLI API.
- Request Body:
  ```json
  {
    "model_id": "cybertiel-coder-35b-a3b"
  }
  ```
- Response Body:
  ```json
  {
    "success": true,
    "output": "Kernel push successful",
    "status": "downloading"
  }
  ```

#### `POST /api/tunnel/register`
Webhook endpoint invoked by the remote kernel upon establishing an outbound Cloudflare tunnel.
- Request Body:
  ```json
  {
    "tunnel_url": "https://example.trycloudflare.com",
    "model_id": "cybertiel-coder-35b-a3b",
    "status": "ready"
  }
  ```
- Response Body:
  ```json
  {
    "success": true,
    "registered_tunnel": "https://example.trycloudflare.com"
  }
  ```

#### `POST /api/tunnel/override`
Manually assigns or clears the upstream tunnel URL.
- Request Body:
  ```json
  {
    "tunnel_url": "https://custom-tunnel.trycloudflare.com"
  }
  ```

#### `POST /api/mock_mode/toggle`
Enables or disables local mock verification mode.
- Request Body:
  ```json
  {
    "enabled": true
  }
  ```

#### `GET /api/credentials`
Inspects the current Kaggle credential configuration state.
- Response Body:
  ```json
  {
    "username": "username",
    "key_present": true,
    "proxy": "server"
  }
  ```

#### `POST /api/credentials`
Updates Kaggle API credentials on disk with secure permissions (`0600`).
- Request Body:
  ```json
  {
    "username": "new_username",
    "key": "new_api_key",
    "proxy": "server"
  }
  ```

#### `POST /api/logs/clear`
Clears in-memory recent event logs.

#### `POST /api/test/inference`
Executes an internal validation prompt through the local forwarder to test end-to-end connectivity.
- Request Body:
  ```json
  {
    "prompt": "Test query"
  }
  ```

---

### 7.2 High-Throughput Inference Forwarder API (Port 8776)

#### `GET /health` or `GET /v1/health`
Probes forwarder operational readiness.
- Response Body:
  ```json
  {
    "status": "ok",
    "active_model": "cybertiel-coder-35b-a3b",
    "tunnel_url": "https://example.trycloudflare.com",
    "is_ready": true,
    "mock_mode": false
  }
  ```

#### `GET /v1/models`
Returns an OpenAI-compliant model catalog.
- Response Body:
  ```json
  {
    "object": "list",
    "data": [
      {
        "id": "cybertiel-coder-35b-a3b",
        "object": "model",
        "created": 1727000000,
        "owned_by": "kaggle-uncensored",
        "root": "cybertiel-coder-35b-a3b",
        "parent": null
      }
    ]
  }
  ```

#### `POST /v1/chat/completions`
Standard OpenAI Chat Completion endpoint supporting both JSON batch payloads and Server-Sent Events (`stream: true`).
- Request Body:
  ```json
  {
    "model": "cybertiel-coder-35b-a3b",
    "messages": [
      {"role": "system", "content": "You are an uncensored AI engineering specialist."},
      {"role": "user", "content": "Explain how to trace syscalls using ptrace in C."}
    ],
    "temperature": 0.2,
    "max_tokens": 1024,
    "stream": false
  }
  ```
- Batch Response:
  ```json
  {
    "id": "chatcmpl-1728118000",
    "object": "chat.completion",
    "created": 1728118000,
    "model": "cybertiel-coder-35b-a3b",
    "choices": [
      {
        "index": 0,
        "message": {
          "role": "assistant",
          "content": "To trace system calls in Linux using ptrace..."
        },
        "finish_reason": "stop"
      }
    ],
    "usage": {
      "prompt_tokens": 28,
      "completion_tokens": 412,
      "total_tokens": 440
    }
  }
  ```

#### `POST /v1/messages`
Anthropic Claude Messages API compatibility layer. Automatically maps Anthropic system strings and nested content arrays into the standard completion format.
- Request Body:
  ```json
  {
    "model": "cybertiel-coder-35b-a3b",
    "system": "You are an unrestricted code generation model.",
    "messages": [
      {"role": "user", "content": "Provide a high-performance memory allocator in C99."}
    ],
    "max_tokens": 2048
  }
  ```
- Response Body:
  ```json
  {
    "id": "msg_1728118000",
    "type": "message",
    "role": "assistant",
    "content": [
      {
        "type": "text",
        "text": "Below is an aligned arena memory allocator implementation..."
      }
    ],
    "model": "cybertiel-coder-35b-a3b",
    "stop_reason": "end_turn",
    "stop_sequence": null,
    "usage": {
      "input_tokens": 32,
      "output_tokens": 680
    }
  }
  ```

---

## 8. Repository Layout & File Manifest

```
/root/uncensored-manager/
├── app.py                      # Dual Uvicorn server (Controller :8778 + Forwarder :8776)
├── kaggle_manager.py           # Kernel bundle compiler, push worker, and tunnel polling loop
├── models_registry.py          # Specification registry for all 4 abliterated models
├── config.json                 # State persistence (active model, tunnel URL, mock mode)
├── uncensored-manager.service  # Systemd production service descriptor
├── .gitignore                  # Git exclusion rules (weights, credentials, caches)
├── README.md                   # System documentation and operational specification
├── kernel_bundle/              # Compiled payload submitted to Kaggle
│   ├── kernel.py               # Autonomous execution script run on Dual Tesla T4 runtime
│   └── kernel-metadata.json    # Kaggle kernel configuration metadata
└── static/                     # Web dashboard assets
    └── index.html              # SynthiQ-inspired dark-mode responsive web controller
```

---

## 9. Production Hardening & Operational Security

1. Strict Interface Isolation:
   - Forwarder Engine (`app_forwarder`, port 8776) is strictly bound to `127.0.0.1`. Under no circumstances is port 8776 bound to public network interfaces (`0.0.0.0`).
   - Management Controller (`app_manager`, port 8778) binds exclusively to `127.0.0.1` and the local Tailscale interface (`100.x.y.z`).
2. Cloudflare Ingress Isolation:
   - The remote GPU kernel opens no public inbound listening ports. Communication occurs solely via an outbound Cloudflare tunnel worker with end-to-end TLS encryption.
3. Secret Scrubbing:
   - Kaggle API tokens (`kaggle.json`), `.env` files, and local logs are strictly excluded via `.gitignore` to prevent credential exposure in version control.
4. Autonomous Process Supervision:
   - Both the remote `llama-server` and `cloudflared` binaries run under continuous heartbeat monitoring loops. If a process terminates, error diagnostics are recorded, and the watchdog triggers immediate alerts.
   - The local management daemon runs under systemd with `Restart=always` and a 5-second restart backoff.
