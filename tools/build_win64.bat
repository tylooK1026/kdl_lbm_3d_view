@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0\.."

echo ============================================================
echo   LBM_post_process Win64 build
echo ============================================================
echo.

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python Launcher was not found.
    echo Install 64-bit Python 3.12 from https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

py -3.12 -c "import struct; raise SystemExit(0 if struct.calcsize('P') == 8 else 1)"
if errorlevel 1 (
    echo [ERROR] 64-bit Python 3.12 was not found.
    pause
    exit /b 1
)

if not exist ".venv-win64-build\Scripts\python.exe" (
    echo [1/6] Creating isolated build environment...
    py -3.12 -m venv .venv-win64-build
    if errorlevel 1 goto :failed
)

call ".venv-win64-build\Scripts\activate.bat"
echo [2/6] Installing build dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto :failed
python -m pip install -r requirements.txt -r requirements-build.txt
if errorlevel 1 goto :failed

echo [3/6] Building standalone application folder...
python -m PyInstaller --noconfirm --clean LBM_post_process.spec
if errorlevel 1 goto :failed

echo [4/6] Running packaged dependency self-test...
"dist\LBM_post_process\LBM_post_process.exe" --self-test
if errorlevel 1 goto :failed

if not exist release mkdir release
echo [5/6] Creating portable ZIP...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "Compress-Archive -Path 'dist\LBM_post_process' -DestinationPath 'release\LBM_post_process_win64_portable.zip' -Force"
if errorlevel 1 goto :failed

set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if exist "%ISCC%" (
    echo [6/6] Creating novice-friendly installer...
    "%ISCC%" "installer\LBM_post_process.iss"
    if errorlevel 1 goto :failed
) else (
    echo [6/6] Inno Setup 6 was not found; installer EXE was skipped.
    echo       Portable ZIP is still ready. Install Inno Setup and rerun to create Setup.exe.
)

echo.
echo Build completed successfully.
echo Output directory: %CD%\release
explorer "%CD%\release"
pause
exit /b 0

:failed
echo.
echo [ERROR] Build failed. Review the messages above.
pause
exit /b 1
