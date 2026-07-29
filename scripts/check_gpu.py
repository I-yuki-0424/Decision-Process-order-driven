import sys
import jax

print("=== JAX Hardware Backend Verification ===")
backend = jax.default_backend()
devices = jax.devices()
print(f"JAX Default Backend: {backend}")
print(f"JAX Available Devices: {devices}")

has_gpu = any("gpu" in str(d).lower() or "cuda" in str(d).lower() for d in devices)

if has_gpu:
    print("SUCCESS: CUDA GPU device detected and active in JAX!")
else:
    print("INFO: JAX currently running on CPU. For GPU acceleration, run inside the Docker CUDA container.")
