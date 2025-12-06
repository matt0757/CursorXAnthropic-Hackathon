# Installation Troubleshooting Guide

## Issue: Failed to build 'pandas' when installing dependencies

This error typically occurs when:
1. Python 3.13 is being used (newer version may lack pre-built wheels)
2. Missing build dependencies on Windows
3. Outdated pip/setuptools

### Solution 1: Upgrade pip and setuptools (Recommended)

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Solution 2: Install packages individually with latest versions

If Solution 1 doesn't work, try installing packages one by one to identify the problematic package:

```bash
pip install --upgrade pip setuptools wheel

# Install core dependencies first
pip install numpy
pip install pandas
pip install scikit-learn

# Then install the rest
pip install -r requirements.txt
```

### Solution 3: Use pre-built wheels (Windows)

Ensure you have the latest pip that can find pre-built wheels:

```bash
python -m pip install --upgrade pip
pip install --only-binary :all: pandas numpy scikit-learn
pip install -r requirements.txt
```

### Solution 4: Install build tools (Windows)

If you need to build from source, install Microsoft C++ Build Tools:

1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Install "Desktop development with C++" workload
3. Retry: `pip install -r requirements.txt`

### Solution 5: Use conda (Alternative)

If pip continues to fail, use conda:

```bash
conda install pandas numpy scikit-learn
pip install fastapi uvicorn streamlit lightgbm plotly pydantic requests python-multipart
```

### Solution 6: Use Python 3.11 or 3.12 (If Python 3.13 issues persist)

Python 3.13 is very new and some packages may not have pre-built wheels yet.

Check your Python version:
```bash
python --version
```

If you have Python 3.13, consider using Python 3.11 or 3.12 which have better package support.

### Quick Test

After installation, verify everything works:

```bash
python -c "import pandas; import numpy; import lightgbm; print('All imports successful!')"
```

### Still Having Issues?

1. Check Python version: `python --version`
2. Check pip version: `pip --version`
3. Upgrade pip: `python -m pip install --upgrade pip`
4. Try installing without version pins: Remove `==` version specifiers temporarily

