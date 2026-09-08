<#
.SYNOPSIS
    Atalho na raiz para treinar todas as NF heads de todas as classes.
#>
param(
    [string] $Classes,
    [string] $Recipe = 'unified',
    [int]    $Epochs,
    [int]    $EvalInterval,
    [switch] $DryRun,
    [string] $EvaluateAfter = 'pixel',
    [switch] $ExportToFinalModels,
    [string] $OutDir
)

$targetScript = Join-Path $PSScriptRoot 'anomaly-detection-pesquisa\differnet\scripts\train\run_train_all_nf_heads.ps1'
if (-not (Test-Path $targetScript)) {
    throw "Script alvo nao encontrado: $targetScript"
}

$params = @{}
if ($Classes) { $params['Classes'] = $Classes }
if ($Recipe) { $params['Recipe'] = $Recipe }
if ($Epochs -gt 0) { $params['Epochs'] = $Epochs }
if ($EvalInterval -gt 0) { $params['EvalInterval'] = $EvalInterval }
if ($DryRun) { $params['DryRun'] = $true }
if ($EvaluateAfter) { $params['EvaluateAfter'] = $EvaluateAfter }
if ($ExportToFinalModels) { $params['ExportToFinalModels'] = $true }
if ($OutDir) { $params['OutDir'] = $OutDir }

& $targetScript @params
