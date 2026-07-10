@echo off
echo ===================================================
echo Checking Prerequisites (Hugging Face CLI)...
echo ===================================================

where hf >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing Hugging Face CLI...
    powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
) else (
    echo Hugging Face CLI is already found.
)

echo.
echo ===================================================
echo Syncing 6GB Model Weights directly to C:\ZImage
echo ===================================================

hf download Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32 --local-dir C:\ZImage

echo.
echo Sync Complete! All folders populated.
pause