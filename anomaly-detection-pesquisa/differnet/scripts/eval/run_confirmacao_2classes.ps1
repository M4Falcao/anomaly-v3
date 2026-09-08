<#
.SYNOPSIS
    Passo 2 do plano: confirma a receita vencedora nas duas classes ainda nao
    estudadas (polymer-insulator-upper-shackle e yoke-suspension).

.DESCRIPTION
    Ate aqui a receita foi validada em 3 classes (vari-grip, lightning-rod-suspension,
    glass-insulator). Este script fecha o protocolo rodando o MESMO estagio de
    retreino + ablation nas duas restantes, para responder tres perguntas:

      P1. 'raw' continua batendo 'per_level_minmax'?   (12/12 heads ate agora)
      P2. TTA por flips continua dando ganho positivo?  (+0.007 a +0.036 ate agora)
      P3. levels='02' continua com regret baixo?        (<= 0.017 nas tres classes)

    Por isso a grade NAO fixa 'raw', 'flips' nem '02': ela os testa contra as
    alternativas. Fatores ja descartados de forma unanime (pad_mode replicate,
    pos_calib, margin > 0, image_score topk) ficam fora para nao inflar o grid.

    O protocolo anti-vies de selecao e o mesmo: --val_fraction 0.4 separa o
    conjunto onde a config e escolhida do conjunto onde ela e medida.

.PARAMETER Classes
    Quais classes rodar. Default: as duas restantes.

.PARAMETER DryRun
    So imprime os comandos, sem executar.

.EXAMPLE
    conda activate anomaly_attentideffernet
    cd C:\Users\teo-s\Documents\GitHub\anomaly-v3\anomaly-detection-pesquisa\differnet
    .\scripts\eval\run_confirmacao_2classes.ps1

.EXAMPLE
    .\scripts\eval\run_confirmacao_2classes.ps1 -Classes yoke-suspension -DryRun

.NOTES
    ARMADILHA DO POWERSHELL: uma lista sem aspas vira array de inteiros e o
    argparse recebe outra coisa. `--level_sets 0,01,012` chega como "0 1 12" e a
    ablation testa conjuntos errados EM SILENCIO. Todas as listas abaixo estao
    entre aspas de proposito. Nao remova.
#>

[CmdletBinding()]
param(
    [string[]] $Classes = @('polymer-insulator-upper-shackle', 'yoke-suspension'),
    [switch]   $DryRun
)

# Nao use 'Stop' global em scripts que chamam executaveis nativos (python):
# no PowerShell 5.1, qualquer stderr (tqdm, warnings) vira NativeCommandError
# e aborta a execucao com Stop. Usamos checagem explicita via $LASTEXITCODE.
$ErrorActionPreference = 'Continue'

# Resolve executavel python: prefere diretamente o caminho do env anomaly_attentideffernet
$knownEnvPy = 'C:\Users\teo-s\.conda\envs\anomaly_attentideffernet\python.exe'
if (Test-Path $knownEnvPy) {
    $pythonExe = $knownEnvPy
} else {
    $pythonExe = 'python'
}

$pythonEnv = & $pythonExe -c "import sys; print(sys.prefix)" 2>$null
Write-Host "Python: $pythonExe ($pythonEnv)"
if ($pythonEnv -notmatch 'anomaly_attentideffernet') {
    Write-Warning "AVISO: O ambiente 'anomaly_attentideffernet' nao parece estar ativo (prefix: $pythonEnv)."
    Write-Warning "Recomenda-se executar: conda activate anomaly_attentideffernet"
}

# Orcamento por classe. yoke-suspension tem 4834 imagens de treino (10x vari-grip)
# e 1253 de teste, entao leva menos epocas e um teto balanceado no conjunto de
# teste; --limit preserva TODAS as 46 anomalias e so subamostra as normais.
#
#   classe                          train  test(good+anom)  area media do defeito
#   polymer-insulator-upper-shackle   935     235 + 82            3.09 %
#   yoke-suspension                  4834    1207 + 46            1.96 %
$Budget = @{
    'polymer-insulator-upper-shackle' = @{ Epochs = 30; EvalInterval = 3; Limit = 0 }
    'yoke-suspension'                 = @{ Epochs = 12; EvalInterval = 2; Limit = 200 }
}

# Grade compartilhada. Ver o docstring para o motivo de cada escolha.
$Grid = @(
    '--stage', 'retrain',
    '--variants', 'baseline,featnorm,featnorm_clamp19',
    '--pads', '112',
    '--pad_modes', 'reflect',
    '--ttas', 'none,flips',
    '--score_modes', 'raw,per_level_minmax',
    '--level_sets', '0,2,02,012',
    '--pos_calibs', 'none',
    '--sigmas', '2,4,8',
    '--margins', '0',
    '--image_scores', 'max',
    '--val_fraction', '0.4',
    '--select_metric', 'pixel_auroc',
    '--seed', '42'
)

$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Push-Location $repo
try {
    $logDir = Join-Path $repo 'analysis\runs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'

    foreach ($cls in $Classes) {
        if (-not $Budget.ContainsKey($cls)) {
            throw "Sem orcamento definido para '$cls'. Adicione uma entrada em `$Budget."
        }
        $b = $Budget[$cls]

        $cmdArgs = @('scripts/eval/ablation_nf_head.py', '--class_name', $cls) + $Grid +
                   @('--epochs', "$($b.Epochs)", '--eval_interval', "$($b.EvalInterval)")
        if ($b.Limit -gt 0) { $cmdArgs += @('--limit', "$($b.Limit)") }

        $log = Join-Path $logDir "confirmacao_${cls}_$stamp.log"

        Write-Host ''
        Write-Host ('=' * 78)
        Write-Host "  $cls  (epochs=$($b.Epochs), eval_interval=$($b.EvalInterval), limit=$(if ($b.Limit) { $b.Limit } else { 'todas' }))"
        Write-Host ('=' * 78)
        Write-Host "python $($cmdArgs -join ' ')"
//
        if ($DryRun) { continue }

        # Executa python nativamente sem pipe 2>&1 (evita NativeCommandError no PS 5.1)
        & $pythonExe @cmdArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "ablation falhou para $cls (exit code $LASTEXITCODE)."
        } else {
            Write-Host "  Concluido com sucesso para $cls."
        }
    }

    if (-not $DryRun) {
        Write-Host ''
        Write-Host 'Confirmacao concluida. Compare held_out.csv de cada run e responda P1/P2/P3.'
        Write-Host 'Depois, avalie em producao com a receita ja embutida como default:'
        Write-Host "  python scripts/eval/evaluate_full_pipeline.py --classes $($Classes -join ',')"
    }
} finally {
    Pop-Location
}
