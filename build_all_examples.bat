@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0build_all_examples.ps1" %*
