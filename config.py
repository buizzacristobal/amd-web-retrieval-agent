"""
Centralized Configuration & AMD Hardware/Endpoint Router.
Handles AMD Developer Cloud (MI300X vLLM), Local AMD ROCm (HIP), and Universal Fallback.
"""

import os
import shutil
import subprocess
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class Settings(BaseModel):
    """Global configuration settings for the AMD Web Retrieval Agent."""
    inference_backend: str = Field(
        default=os.getenv("INFERENCE_BACKEND", "universal_fallback"),
        description="Active backend: amd_cloud_vllm, local_rocm, or universal_fallback"
    )

    # Mode 1: AMD Developer Cloud (Instinct MI300X vLLM)
    amd_base_url: str = Field(
        default=os.getenv("AMD_BASE_URL", "https://cloud.amd.com/v1"),
        description="Endpoint for AMD Developer Cloud OpenAI-compatible vLLM instance"
    )
    amd_api_key: str = Field(
        default=os.getenv("AMD_API_KEY", ""),
        description="API Key for AMD Developer Cloud"
    )
    amd_model: str = Field(
        default=os.getenv("AMD_MODEL", "Qwen/Qwen2.5-72B-Instruct"),
        description="Primary model running on AMD Instinct MI300X"
    )

    # Mode 2: Local AMD ROCm (HIP Runtime)
    rocm_device: str = Field(
        default=os.getenv("ROCM_DEVICE", "cuda:0"),
        description="Target HIP device for ROCm local execution"
    )
    rocm_model: str = Field(
        default=os.getenv("ROCM_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
        description="Local ROCm model name"
    )

    # Mode 3: Universal Fallback (Local Ollama, vLLM, or Zero-Setup Extractive)
    openai_compatible_base_url: str = Field(
        default=os.getenv("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000/v1"),
        description="Fallback OpenAI-compatible endpoint URL"
    )
    openai_compatible_api_key: str = Field(
        default=os.getenv("OPENAI_COMPATIBLE_API_KEY", ""),
        description="Fallback API key"
    )
    openai_compatible_model: str = Field(
        default=os.getenv("OPENAI_COMPATIBLE_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        description="Fallback model identifier"
    )

    # Agent Tuning & Guardrails
    max_search_results: int = Field(default=int(os.getenv("MAX_SEARCH_RESULTS", "5")))
    max_retrieval_hops: int = Field(default=int(os.getenv("MAX_RETRIEVAL_HOPS", "2")))
    grounding_strictness_threshold: float = Field(default=float(os.getenv("GROUNDING_STRICTNESS_THRESHOLD", "0.35")))
    request_timeout_seconds: int = Field(default=int(os.getenv("REQUEST_TIMEOUT_SECONDS", "6")))
    user_agent_rotation: bool = Field(default=os.getenv("USER_AGENT_ROTATION", "true").lower() == "true")

def detect_rocm_runtime() -> Dict[str, Any]:
    """Detect whether local AMD ROCm HIP runtime or ROCm tools are available."""
    info = {
        "hip_available": False,
        "hip_version": None,
        "device_count": 0,
        "device_name": None,
        "rocm_smi_available": False,
        "rocm_path_exists": os.path.exists("/opt/rocm")
    }

    # Check for rocm-smi command
    if shutil.which("rocm-smi"):
        info["rocm_smi_available"] = True

    # Check PyTorch HIP support if torch is installed
    try:
        import torch
        if hasattr(torch, "version") and hasattr(torch.version, "hip") and torch.version.hip is not None:
            info["hip_available"] = True
            info["hip_version"] = torch.version.hip
            if torch.cuda.is_available():
                info["device_count"] = torch.cuda.device_count()
                info["device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["runtime_note"] = str(exc)

    return info

def get_hardware_target(backend: str) -> str:
    """Return descriptive hardware architecture target for UI & Telemetry HUD."""
    rocm_info = detect_rocm_runtime()
    if backend == "amd_cloud_vllm":
        return "AMD Instinct MI300X (vLLM / AMD Developer Cloud)"
    elif backend == "local_rocm":
        if rocm_info["hip_available"]:
            dev = rocm_info["device_name"] or "AMD HIP GPU"
            hip_ver = rocm_info["hip_version"] or "6.x"
            return f"{dev} (ROCm {hip_ver} / PyTorch HIP)"
        return "AMD ROCm HIP Emulation (Host CPU Runtime)"
    else:
        # universal_fallback
        return "AMD Ryzen / Host Engine (Extractive Grounding)"

def get_vram_utilization_est() -> str:
    """Estimate GPU VRAM or system RAM utilization for telemetry."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
            return f"{allocated:.1f}GB / {reserved:.1f}GB VRAM"
    except Exception as exc:
        _ = exc

    # Fallback to system RAM via psutil if available
    try:
        import psutil
        mem = psutil.virtual_memory()
        return f"{mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB Host RAM ({mem.percent}%)"
    except Exception:
        return "N/A (Standard Host Runtime)"

# Global settings instance
_settings_instance = None

def get_settings() -> Settings:
    """Singleton getter for application settings."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance
