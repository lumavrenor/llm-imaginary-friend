@echo off
rem Start llm-imaginary-friend. Extra args pass through (e.g. go.bat --no-idle).
rem   go.bat             Japanese edition (config.toml, VOICEVOX)
rem   go.bat en          English edition (config.en.toml, Kokoro TTS on the CPU)
rem   go.bat en --cli    remaining args pass through to imaginary_friend.main
cd /d "%~dp0"
set "LANG_ARGS="
set "NEED_EN=0"
if /i "%~1"=="en" (
    set "LANG_ARGS=--config config.en.toml"
    set "NEED_EN=1"
    shift
)
if /i "%~1"=="ja" shift
if not exist ".venv\Scripts\python.exe" (
    echo [first-time setup] creating .venv ...
    python -m venv .venv || exit /b 1
    ".venv\Scripts\python.exe" -m pip install -q -r requirements.txt || exit /b 1
)
if "%NEED_EN%"=="1" (
    ".venv\Scripts\python.exe" -c "import onnxruntime, misaki, g2p_en, en_core_web_sm" >nul 2>&1
    if errorlevel 1 (
        echo [first-time setup] installing the English edition packages ^(requirements-en.txt^) ...
        rem a .venv made by uv has no pip ^(spacy download also needs pip^), so add it first
        ".venv\Scripts\python.exe" -m pip --version >nul 2>&1 || ".venv\Scripts\python.exe" -m ensurepip --upgrade || exit /b 1
        ".venv\Scripts\python.exe" -m pip install -q -r requirements-en.txt || exit /b 1
        ".venv\Scripts\python.exe" -m spacy download en_core_web_sm || exit /b 1
    )
)
rem "shift" does not change %*, so rebuild the remaining args
set "REST="
:collect
if "%~1"=="" goto run
set REST=%REST% %1
shift
goto collect
:run
".venv\Scripts\python.exe" -m imaginary_friend.main %LANG_ARGS% %REST%
