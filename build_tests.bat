@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0build_tests.ps1" %*
