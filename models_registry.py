"""
Model Registry for Autonomous Kaggle Uncensored Infrastructure.
Contains verified abliterated GGUF models.
Using -sm layer (pipeline parallelism) which works 100% on Dual T4 CUDA.
Maximized context windows via YaRN RoPE scaling and q4_0 KV caching.
"""

MODELS = {
    "cybertiel-coder-35b-a3b": {
        "id": "cybertiel-coder-35b-a3b",
        "name": "CyberTiel-Coder-35B-A3B (Abliterated MoE)",
        "repo_id": "peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF",
        "filename": "Cyber-Tiel-Coder-35B-A3B-UD-Q4_K_XL.gguf",
        "parameters": "35B MoE (~3.4B active)",
        "quant": "UD-Q4_K_XL (Cyber-imatrix)",
        "filesize_gb": 22.36,
        "context_length": 262144,
        "context_display": "262,144 tokens (262k / ~2.75 lakh)",
        "rope_scaling": "YaRN scale 8 (orig 32k)",
        "kv_cache": "q4_0 / q4_0",
        "vram_requirement": "24.8 GB / 32 GB (Dual T4)",
        "speed": "40-55 tok/s (MoE Dual T4)",
        "rank": 1,
        "description": "State-of-the-art offensive/defensive cybersecurity & agentic coding model. Zero refusals on HarmBench (all categories). Calibrated on Exploit-DB, HackTricks, PayloadsAllTheThings and Cybench CTFs. Scaled to 262k context (~2.75 lakh) via YaRN RoPE scale 8 with q4_0 KV cache.",
        "server_args": [
            "--rope-scaling", "yarn",
            "--rope-scale", "8",
            "--yarn-orig-ctx", "32768",
            "-c", "262144",
            "-ctk", "q4_0",
            "-ctv", "q4_0",
            "-ngl", "52",
            "-sm", "layer",
            "-ts", "1,1",
            "-b", "2048",
            "-ub", "512",
            "-fa", "auto",
            "--threads", "4"
        ]
    },
    "qwen3.8-27b-abliterated": {
        "id": "qwen3.8-27b-abliterated",
        "name": "Qwen3.8-27B-Instruct (Abliterated RVN)",
        "repo_id": "cole17e/Qwen3.8-27B-Heretic-Abliterated-Uncensored-GGUF",
        "filename": "RVN-Q4_K_M-multilingual.gguf",
        "parameters": "27B Dense (Gen 3.8)",
        "quant": "Q4_K_M",
        "filesize_gb": 16.55,
        "context_length": 262144,
        "context_display": "262,144 tokens (262k)",
        "rope_scaling": "YaRN scale 8 (orig 32k)",
        "kv_cache": "q4_0 / q4_0",
        "vram_requirement": "22.5 GB / 32 GB (Dual T4)",
        "speed": "28-35 tok/s (Dual T4)",
        "rank": 2,
        "description": "Latest Generation Alibaba Qwen3.8 model. ARA-refined 3x triple-pass abliterated (0 refusals on security tasks). Scaled to 262k context via YaRN RoPE scale 8 with q4_0 KV cache on Dual T4.",
        "server_args": [
            "--rope-scaling", "yarn",
            "--rope-scale", "8",
            "--yarn-orig-ctx", "32768",
            "-c", "262144",
            "-ctk", "q4_0",
            "-ctv", "q4_0",
            "-ngl", "58",
            "-sm", "layer",
            "-ts", "1,1",
            "-b", "2048",
            "-ub", "512",
            "-fa", "auto",
            "--threads", "4"
        ]
    },
    "qwen2.5-coder-32b-abliterated": {
        "id": "qwen2.5-coder-32b-abliterated",
        "name": "Qwen2.5-Coder-32B-Instruct (Abliterated)",
        "repo_id": "bartowski/Qwen2.5-Coder-32B-Instruct-abliterated-GGUF",
        "filename": "Qwen2.5-Coder-32B-Instruct-abliterated-Q4_K_M.gguf",
        "parameters": "32.5B Dense",
        "quant": "Q4_K_M",
        "filesize_gb": 19.85,
        "context_length": 131072,
        "context_display": "131,072 tokens (131k)",
        "rope_scaling": "YaRN scale 4 (orig 32k)",
        "kv_cache": "q4_0 / q4_0",
        "vram_requirement": "26.5 GB / 32 GB (Dual T4)",
        "speed": "20-25 tok/s (Dual T4)",
        "rank": 3,
        "description": "Specialized coding powerhouse with refusal vector ablation. HumanEval ~90%. Scaled to 131k context via YaRN RoPE scale 4 with q4_0 KV cache on Dual T4.",
        "server_args": [
            "--rope-scaling", "yarn",
            "--rope-scale", "4",
            "--yarn-orig-ctx", "32768",
            "-c", "131072",
            "-ctk", "q4_0",
            "-ctv", "q4_0",
            "-ngl", "52",
            "-sm", "layer",
            "-ts", "1,1",
            "-b", "2048",
            "-ub", "512",
            "-fa", "auto",
            "--threads", "4"
        ]
    },
    "llama-3.3-70b-abliterated": {
        "id": "llama-3.3-70b-abliterated",
        "name": "Llama-3.3-70B-Instruct (Abliterated)",
        "repo_id": "bartowski/Llama-3.3-70B-Instruct-abliterated-GGUF",
        "filename": "Llama-3.3-70B-Instruct-abliterated-IQ3_M.gguf",
        "parameters": "70.6B Dense",
        "quant": "IQ3_M",
        "filesize_gb": 28.5,
        "context_length": 65536,
        "context_display": "65,536 tokens (65k YaRN)",
        "rope_scaling": "YaRN (scale 2, orig 32k)",
        "kv_cache": "q4_0 / q4_0",
        "vram_requirement": "31.5 GB / 32 GB (Dual T4)",
        "speed": "8-12 tok/s (Dual T4)",
        "rank": 4,
        "description": "Maximum parameter reasoning giant. IQ3_M quant with scaled 65k context and q4_0 KV cache across Dual T4 VRAM.",
        "server_args": [
            "--rope-scaling", "yarn",
            "--rope-scale", "2",
            "--yarn-orig-ctx", "32768",
            "-c", "65536",
            "-ctk", "q4_0",
            "-ctv", "q4_0",
            "-ngl", "40",
            "-sm", "layer",
            "-ts", "1,1",
            "-b", "1024",
            "-ub", "256",
            "-fa", "auto",
            "--threads", "4"
        ]
    }
}

AVAILABLE_MODELS = MODELS
DEFAULT_MODEL_ID = "cybertiel-coder-35b-a3b"
