@echo off
REM Ensure WSL is available
wsl --help >nul 2>&1
IF ERRORLEVEL 1 (
    echo WSL is not installed. Please install WSL before proceeding.
    exit /b 1
)

REM Connect to the virtual environment
wsl bash -c "source ./ic-light/bin/activate"
IF ERRORLEVEL 1 (
    echo An error occurred while activating the virtual environment.
    exit /b 1
)

REM Run the Python program
REM Replace 'gradio_demo.py' with the actual script name
wsl bash -c "./ic-light/bin/python ./gradio_demo_bg.py"
IF ERRORLEVEL 1 (
    echo An error occurred while running the program.
    exit /b 1
)

REM Keep the terminal open
cmd /k
