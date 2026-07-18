@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title LBM_post_process - 一键生成 Win64 软件

echo ============================================================
echo   LBM_post_process V1.0.0 - 本地一键生成 Win64 软件
echo ============================================================
echo.
echo 本脚本不使用 GitHub。
echo 它会自动准备 Python 3.12、构建环境和 Inno Setup，
echo 最后生成安装版 EXE 和便携版 ZIP。
echo.
echo 首次运行需要联网下载约数百 MB 文件，并需要较多磁盘空间。
echo 整个过程通常需要 10 到 30 分钟，请不要关闭黑色窗口。
echo.
pause

if /i not "%PROCESSOR_ARCHITECTURE%"=="AMD64" if /i not "%PROCESSOR_ARCHITEW6432%"=="AMD64" (
    echo [错误] 当前系统不是受支持的 Windows x64 环境。
    echo 本构建脚本适用于 Windows 10/11 64 位。
    goto :failed
)

where winget >nul 2>nul
if errorlevel 1 (
    echo [错误] Windows 中没有找到 winget（应用安装程序）。
    echo 即将打开微软商店，请安装或更新“应用安装程序 App Installer”，
    echo 安装完成后重新双击本文件。
    start "" "ms-windows-store://pdp/?ProductId=9NBLGGH4NNS1"
    goto :failed
)

call :find_python
if not defined PYTHON_EXE (
    echo [1/8] 正在自动安装 Python 3.12 x64，仅用于生成软件...
    winget install --id Python.Python.3.12 --exact --source winget --scope user --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [错误] Python 自动安装失败。
        goto :failed
    )
    call :find_python
)

if not defined PYTHON_EXE (
    echo [错误] Python 已安装，但脚本仍无法找到它。请重启电脑后再试。
    goto :failed
)

"%PYTHON_EXE%" -c "import struct; raise SystemExit(0 if struct.calcsize('P') == 8 else 1)"
if errorlevel 1 (
    echo [错误] 检测到的 Python 不是 64 位版本。
    goto :failed
)

echo [2/8] 正在创建隔离构建环境...
if not exist ".venv-win64-build\Scripts\python.exe" (
    "%PYTHON_EXE%" -m venv .venv-win64-build
    if errorlevel 1 goto :failed
)
set "BUILD_PYTHON=%CD%\.venv-win64-build\Scripts\python.exe"

echo [3/8] 正在安装软件构建组件，请耐心等待...
"%BUILD_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%BUILD_PYTHON%" -m pip install --only-binary=:all: -r requirements.txt -r requirements-build.txt
if errorlevel 1 goto :failed

echo [4/8] 正在生成 LBM_post_process 独立程序...
"%BUILD_PYTHON%" -m PyInstaller --noconfirm --clean LBM_post_process.spec
if errorlevel 1 goto :failed

echo [5/8] 正在执行程序自检...
"dist\LBM_post_process\LBM_post_process.exe" --self-test
if errorlevel 1 goto :failed

if not exist release mkdir release
echo [6/8] 正在生成便携版 ZIP...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path 'dist\LBM_post_process' -DestinationPath 'release\LBM_post_process_win64_portable.zip' -Force"
if errorlevel 1 goto :failed

call :find_iscc
if not defined ISCC_EXE (
    echo [7/8] 正在安装安装包制作工具 Inno Setup...
    echo Windows 可能弹出权限确认窗口，请点击“是”。
    winget install --id JRSoftware.InnoSetup --exact --source winget --silent --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [警告] Inno Setup 自动安装失败，便携版 ZIP 已经生成。
    )
    call :find_iscc
) else (
    echo [7/8] 已找到 Inno Setup。
)

if defined ISCC_EXE (
    echo [8/8] 正在生成小白安装版 EXE...
    "%ISCC_EXE%" "installer\LBM_post_process.iss"
    if errorlevel 1 goto :failed
) else (
    echo [警告] 未生成安装版 EXE，但便携版 ZIP 可以直接使用。
)

echo.
echo ============================================================
echo   生成完成
echo ============================================================
echo.
echo 输出位置：
echo %CD%\release
echo.
echo 安装版：LBM_post_process_Setup_win64.exe
echo 便携版：LBM_post_process_win64_portable.zip
echo.
start "" explorer "%CD%\release"
pause
exit /b 0

:find_python
set "PYTHON_EXE="
if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python312\python.exe"
if defined PYTHON_EXE exit /b 0
for /f "usebackq delims=" %%P in (`py -3.12 -c "import sys; print(sys.executable)" 2^>nul`) do set "PYTHON_EXE=%%P"
exit /b 0

:find_iscc
set "ISCC_EXE="
if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if exist "%LocalAppData%\Programs\Inno Setup 6\ISCC.exe" set "ISCC_EXE=%LocalAppData%\Programs\Inno Setup 6\ISCC.exe"
exit /b 0

:failed
echo.
echo ============================================================
echo   未能完成生成
echo ============================================================
echo 请保留此窗口中的错误信息并截图发给我，我可以继续帮你处理。
echo.
pause
exit /b 1
