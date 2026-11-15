@echo off
rem Delete the conversation database (shared log, memories, moods, relationships).
rem   reset.bat                  delete the DB set in config.toml ([store] path)
rem   reset.bat --all            delete every *.db in data\ (old DBs too)
rem   reset.bat --config FILE    read [store] path from FILE instead of config.toml
rem   reset.bat -y               do not ask for confirmation
rem Close the app before running this.
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "ALL=0"
set "YES=0"
set "CONFIG=config.toml"
:parse
if "%~1"=="" goto parsed
if /i "%~1"=="--all"    ( set "ALL=1" & shift & goto parse )
if /i "%~1"=="-y"       ( set "YES=1" & shift & goto parse )
if /i "%~1"=="--yes"    ( set "YES=1" & shift & goto parse )
if /i "%~1"=="--config" ( set "CONFIG=%~2" & shift & shift & goto parse )
echo unknown option: %~1
exit /b 2
:parsed

if exist ".venv\Scripts\python.exe" (set "PY=.venv\Scripts\python.exe") else (set "PY=python")

set "COUNT=0"
if "%ALL%"=="1" goto collect_all

for /f "usebackq delims=" %%P in (`call "%PY%" -c "import sys, tomllib; print(tomllib.load(open(sys.argv[1], 'rb')).get('store', {}).get('path', 'data/imaginary_friend.db'))" "%CONFIG%"`) do set "DB=%%P"
if not defined DB goto read_error
set "DB=!DB:/=\!"
if exist "!DB!" (
    set "COUNT=1"
    set "T1=!DB!"
)
goto collected

:read_error
echo could not read [store] path from %CONFIG% (needs Python 3.11+)
exit /b 1

:collect_all
for %%F in (data\*.db) do (
    set /a COUNT+=1
    set "T!COUNT!=%%F"
)
:collected

if "%COUNT%"=="0" (
    echo nothing to delete.
    exit /b 0
)

echo will delete:
for /l %%i in (1,1,%COUNT%) do (
    for %%S in ("" "-wal" "-shm" "-journal") do (
        if exist "!T%%i!%%~S" echo   !T%%i!%%~S
    )
)

if not "%YES%"=="1" (
    set "ANS="
    set /p "ANS=really delete? [y/N] "
    if /i not "!ANS!"=="y" if /i not "!ANS!"=="yes" (
        echo cancelled.
        exit /b 1
    )
)

set "FAIL=0"
for /l %%i in (1,1,%COUNT%) do (
    for %%S in ("" "-wal" "-shm" "-journal") do (
        if exist "!T%%i!%%~S" del /f /q "!T%%i!%%~S" || set "FAIL=1"
        if exist "!T%%i!%%~S" set "FAIL=1"
    )
)
if "%FAIL%"=="1" (
    echo some files could not be deleted. is the app still running?
    exit /b 1
)
echo done. a fresh DB will be created on next start.
