<#
.SYNOPSIS
    Atalho na raiz para executar a confirmacao das duas classes restantes.
#>
param(
    [string[]] $Classes = @('polymer-insulator-upper-shackle', 'yoke-suspension'),
    [switch]   $DryRun
)

$targetScript = Join-Path $PSScriptRoot 'anomaly-detection-pesquisa\differnet\scripts\eval\run_confirmacao_2classes.ps1'
if (-not (Test-Path $targetScript)) {
    throw "Script alvo nao encontrado: $targetScript"
}

$params = @{ Classes = $Classes }
if ($DryRun) { $params['DryRun'] = $true }

& $targetScript @params
