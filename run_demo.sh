#!/usr/bin/env bash
set -e

# Navigate to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "=========================================================="
echo "⚡ AMD Intelligent Web Retrieval Agent (Match 4)"
echo "Directory: $PROJECT_DIR"
echo "=========================================================="

# 1. Run Verification Pipeline
echo -e "\n[1/2] Running Automated Verification Benchmark Suite..."
python3 verify_pipeline.py

# 2. Check for Streamlit
echo -e "\n[2/2] Checking Streamlit UI..."
if command -v streamlit &> /dev/null; then
    echo "Starting Streamlit UI on http://localhost:8501 ..."
    streamlit run app.py
elif [ -f ".venv/bin/streamlit" ]; then
    echo "Starting Streamlit from .venv on http://localhost:8501 ..."
    .venv/bin/streamlit run app.py
else
    echo -e "\n⚠️  Streamlit is not yet installed in your current environment."
    echo "To launch the graphical dashboard, run:"
    echo "    cd $PROJECT_DIR"
    echo "    source .venv/bin/activate"
    echo "    pip install -r requirements.txt"
    echo "    streamlit run app.py"
fi
