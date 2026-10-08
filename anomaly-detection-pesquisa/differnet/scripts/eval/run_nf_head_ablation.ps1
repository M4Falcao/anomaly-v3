[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$DatasetRoot,

    [string[]]$Classes = @(
        'glass-insulator',
        'lightning-rod-suspension',
        'polymer-insulator-upper-shackle',
        'vari-grip',
        'yoke-suspension'
    ),

    [int[]]$Seeds = @(42),
    [int]$SplitSeed = 42,
    [int]$Bootstrap = 1000,
    [string]$OutputRoot,
    [string]$Python = 'python',
    [int]$TopN = 10,
    [switch]$PlanOnly
)

$ErrorActionPreference = 'Stop'
$DatasetRoot = (Resolve-Path $DatasetRoot).Path
$AblationScript = Join-Path $PSScriptRoot 'ablation_nf_head.py'
$KnownClasses = @(
    'glass-insulator',
    'lightning-rod-suspension',
    'polymer-insulator-upper-shackle',
    'vari-grip',
    'yoke-suspension'
)
$EpochBudgets = @{
    'glass-insulator' = @(40, 5)
    'lightning-rod-suspension' = @(40, 5)
    'polymer-insulator-upper-shackle' = @(30, 3)
    'vari-grip' = @(40, 5)
    'yoke-suspension' = @(12, 2)
}

if (-not (Test-Path $AblationScript -PathType Leaf)) {
    throw "Ablation script not found: $AblationScript"
}

foreach ($className in $Classes) {
    if ($className -notin $KnownClasses) {
        throw "Unknown class '$className'. Valid classes: $($KnownClasses -join ', ')"
    }
    foreach ($relativePath in @("$className\train\good", "$className\test\good", "$className\ground_truth")) {
        $requiredPath = Join-Path $DatasetRoot $relativePath
        if (-not (Test-Path $requiredPath -PathType Container)) {
            throw "Dataset layout is incomplete: $requiredPath"
        }
    }
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $OutputRoot = Join-Path $projectRoot "resultado_analise_final\ablacao_nf_head\run_full_$stamp"
}
if (-not $PlanOnly) {
    New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null
}

Write-Warning 'Validation/holdout are partitions of the existing labeled test split, not an external test. Do not select the final recipe using holdout results.'
Write-Warning 'This runner trains new NF heads from random initialization for every selected variant and seed; it does not load released NF heads.'
Write-Warning 'The SEDifferNet checkpoint remains pretrained and frozen. This is the local head ablation, not an end-to-end CFLOW-AD architecture reproduction.'
Write-Host "Dataset: $DatasetRoot"
Write-Host "Output:  $OutputRoot"
Write-Host "Training seeds: $($Seeds -join ',') | split seed: $SplitSeed (fixed across seeds)"

foreach ($className in $Classes) {
    $budget = $EpochBudgets[$className]
    $runDir = Join-Path $OutputRoot $className
    $arguments = @(
        $AblationScript,
        '--class_name', $className,
        '--dataset', $DatasetRoot,
        '--stage', 'retrain',
        '--variants', 'baseline,featnorm,featnorm_l1_only,clamp19,featnorm_clamp19',
        '--epochs', [string]$budget[0],
        '--eval_interval', [string]$budget[1],
        '--train_score_mode', 'raw',
        '--val_fraction', '0.4',
        '--train_seeds', ($Seeds -join ','),
        '--split_seed', [string]$SplitSeed,
        '--out_dir', $runDir,
        '--pads', '0,112',
        '--pad_modes', 'reflect,replicate',
        '--ttas', 'none,hflip,flips',
        '--score_modes', 'cflow_original,raw,per_level_minmax,per_level_prob,per_level_std',
        '--align_corners_modes', 'true,false',
        '--level_sets', '0,1,2,01,02,12,012',
        '--pos_calibs', 'none,l23',
        '--sigmas', '0,2,4,8,12',
        '--margins', '0',
        '--image_scores', 'max,topk',
        '--select_metric', 'auroc_aupro',
        '--bootstrap', [string]$Bootstrap,
        '--top_n', [string]$TopN,
        '--pixel_stride', '2'
    )

    Write-Host "`n$className"
    if ($PlanOnly) {
        Write-Host ($Python + ' ' + ($arguments -join ' '))
        continue
    }
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Ablation failed for $className (exit code $LASTEXITCODE)."
    }
}