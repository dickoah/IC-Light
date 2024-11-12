@echo off

REM Check if WSL is available
wsl --help >nul 2>&1
IF ERRORLEVEL 1 (
    echo ============================================================
    echo WSL (Windows Subsystem for Linux) is not installed.
    echo Please install WSL by following these steps:
    echo 1. Open PowerShell as Administrator.
    echo 2. Run: wsl --install
    echo 3. Restart your computer.
    echo For more details, visit: https://learn.microsoft.com/en-us/windows/wsl/install
    echo ============================================================
    exit /b 1
)

REM Ensure python3-venv is installed
echo Checking if python3-venv is installed...
wsl bash -c "dpkg -l | grep -q python3-venv"
IF ERRORLEVEL 1 (
    echo python3-venv is not installed. Installing it now...
    wsl bash -c "sudo apt update && sudo apt install -y python3-venv"
    IF ERRORLEVEL 1 (
        echo An error occurred while installing python3-venv.
        exit /b 1
    )
)

REM Create the virtual environment
wsl bash -c "python3 -m venv ./ic-light"
IF ERRORLEVEL 1 (
    echo An error occurred while creating the virtual environment.
    exit /b 1
)

REM Activate the virtual environment
wsl bash -c "source ./ic-light/bin/activate"
IF ERRORLEVEL 1 (
    echo An error occurred while activating the virtual environment.
    exit /b 1
)

REM Upgrade pip
wsl bash -c "./ic-light/bin/pip install -U pip"
IF ERRORLEVEL 1 (
    echo An error occurred while updating pip.
    exit /b 1
)

REM Install dependencies
wsl bash -c "./ic-light/bin/pip install -r ./requirements.txt"
IF ERRORLEVEL 1 (
    echo An error occurred while installing the requirements.
    exit /b 1
)

echo ============================================================
echo Virtual environment 'ic-light' created and requirements installed successfully!
echo ============================================================
pause
