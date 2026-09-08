<#
.SYNOPSIS
    Executa o treinamento de todas as NF Heads para todas as classes do dataset.

.DESCRIPTION
    Wrapper PowerShell para scripts/train/train_all_nf_heads.py.
    Garante o uso do interpretador Python correto (ambiente anomaly_attentideffernet)
    e evita NativeCommandError no PowerShell 5.1.

.PARAMETER Classes
    Lista de classes a treinar separadas por virgula (default: todas as 5).

.PARAMETER Recipe
    Receita de treino: unified (default), selective_l1, optimal, baseline.

.PARAMETER Epochs
    Numero de epocas (se omitido, usa os orcamentos por classe da receita).

.PARAMETER DryRun
    Apenas exibe os comandos e configuracoes sem executar.

.PARAMETER EvaluateAfter
    Avaliacao pos-treino: pixel (default), full, none.

.PARAMETER ExportToFinalModels
    Copia os melhores modelos gerados para final_models/NF Head/<classe>/best_models/ com backup.

.EXAMPLE
    .\scripts\train\run_train_all_nf_heads.ps1 -DryRun
    .\scripts\train\run_train_all_nf_heads.ps1
    .\scripts\train\run_train_all_nf_heads.ps1 -Classes "vari-grip,glass-insulator" -Epochs 20
    .\scripts\train\run_train_all_nf_heads.ps1 -Recipe selective_l1 -Epochs 80
#>

[CmdletBinding()]
param(
    [string] $Classes,
    [ValidateSet('unified', 'selective_l1', 'optimal', 'baseline')]
    [string] $Recipe = 'unified',
    [int]    $Epochs,
    [int]    $EvalInterval,
    [switch] $DryRun,
    [ValidateSet('pixel', 'full', 'none')]
    [string] $EvaluateAfter = 'pixel',
    [switch] $ExportToFinalModels,
    [string] $OutDir
)

$ErrorActionPreference = 'Continue'

# Localiza interpretador Python do ambiente Conda
$knownEnvPy = 'C:\Users\teo-s\.conda\envs\anomaly_attentideffernet\python.exe'
if (Test-Path $knownEnvPy) {
    $pythonExe = $knownEnvPy
} else {
    $pythonExe = 'python'
}

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $repo
try {
    $cmdArgs = @('scripts/train/train_all_nf_heads.py', '--recipe', $Recipe)

    if ($Classes) {
        $cmdArgs += @('--classes', $Classes)
    }
    if ($Epochs -gt 0) {
        $cmdArgs += @('--epochs', "$Epochs")
    }
    if ($EvalInterval -gt 0) {
        $cmdArgs += @('--eval_interval', "$EvalInterval")
    }
    if ($DryRun) {
        $cmdArgs += @('--dry_run')
    }
    if ($EvaluateAfter) {
        $cmdArgs += @('--evaluate_after', $EvaluateAfter)
    }
    if ($ExportToFinalModels) {
        $cmdArgs += @('--export_to_final_models')
    }
    if ($OutDir) {
        $cmdArgs += @('--out_dir', $OutDir)
    }

    Write-Host "Executando: $pythonExe $($cmdArgs -join ' ')"
    & $pythonExe @cmdArgs
} finally {
    Pop-Location
}
