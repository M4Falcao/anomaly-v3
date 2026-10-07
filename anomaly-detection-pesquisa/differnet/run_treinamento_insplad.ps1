# Script PowerShell para iniciar o treinamento em lote do SEDifferNet em todas as classes

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

Write-Host "Iniciando o treinamento do SEDifferNet em lote (todas as classes do INSPLAD)..." -ForegroundColor Cyan
python train_insplad.py
