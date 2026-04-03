@echo off
setlocal
chcp 65001 >nul

cd /d %~dp0
set "NO_PAUSE=0"
if /I "%~1"=="/nopause" set "NO_PAUSE=1"

echo ========================================
echo Medical QA Real Model Demo
echo Model: stepfun-ai/Step-3.5-Flash
echo ========================================
echo.

rem Try to activate conda base for double-click runs from Explorer.
where conda >nul 2>nul
if not errorlevel 1 (
  call conda activate base >nul 2>nul
)

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found in PATH.
  goto :fail_pause
)

set "PYTHONPATH=%cd%;%PYTHONPATH%"

echo [1/2] Real API smoke test (question_creator)...
python -m tests.manual.test_question_creator_manual --model stepfun-ai/Step-3.5-Flash --save
if errorlevel 1 goto :fail_pause
echo.

echo [2/2] Real API core role test (facet_qa_agent, parser/validator strict)...
python -m tests.manual.test_facet_qa_agent_manual --model stepfun-ai/Step-3.5-Flash --save
if errorlevel 1 (
  echo [WARN] facet_qa_agent strict format validation failed.
  echo [WARN] This means model output format may need stronger prompt constraints, but API is reachable.
)

echo.
echo ========================================
echo Real model demo finished.
echo ========================================
goto :success_end

:fail_pause
echo.
echo ========================================
echo Real model demo failed.
echo ========================================
echo.
if "%NO_PAUSE%"=="0" pause
endlocal
exit /b 1

:success_end
echo.
if "%NO_PAUSE%"=="0" pause
endlocal
