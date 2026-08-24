#!/usr/bin/env bash
# Run this ON the RunPod pod, from inside the DADS project directory (e.g. /workspace/DADS),
# after the project files have been transferred there. Sets up a venv INSIDE /workspace so it
# survives Stop/Start, and swaps the CPU-only onnxruntime for onnxruntime-gpu so the ONNX leg
# of the benchmark actually uses the pod's GPU instead of falling back to CPU.
set -e

if [ ! -d /workspace/venv ]; then
    python3 -m venv /workspace/venv
fi
source /workspace/venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

# requirements.txt pins the CPU-only onnxruntime package; the GPU one has to replace it,
# not sit alongside it — same import name, so both installed together is unsupported.
pip uninstall -y onnxruntime || true
pip install onnxruntime-gpu

echo ""
echo "Setup complete."
echo "Activate this environment in future sessions with: source /workspace/venv/bin/activate"
echo ""
which trtexec >/dev/null 2>&1 && echo "trtexec found — TensorRT leg will run." \
    || echo "trtexec NOT found — pick a pod template with TensorRT preinstalled, or the TensorRT leg will be skipped."
