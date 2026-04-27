@echo off
REM Duality AI Hackathon - Environment Setup (Windows)
REM Run in Anaconda Prompt: setup_env.bat

SET ENV_NAME=EDU

echo ======================================================
echo   Duality AI Segmentation - Environment Setup
echo   Creating conda environment: %ENV_NAME%
echo ======================================================

call conda create -y -n %ENV_NAME% python=3.10
call conda activate %ENV_NAME%

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy Pillow matplotlib tqdm scikit-learn opencv-python-headless

echo.
echo ======================================================
echo   Setup complete!
echo   Activate: conda activate %ENV_NAME%
echo   Train:    python train.py
echo ======================================================
pause
