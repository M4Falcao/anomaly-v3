<#
.SYNOPSIS
  Ablação de máscara de objeto (D-19): máscaras YOLO26 -> audit -> poda -> NF head mascarada -> relatório.

.DESCRIPTION
  Etapas (-Stage):
    convert  01  dataset sem fundo -> YOLO-seg            (CPU)
    train    02  treina YOLO26-seg por classe              (GPU)
    masks    03  máscaras no dataset original              (GPU; -MaskSource nobg = CPU)
    audit    04  área mantida × GT perdido por estratégia  (CPU) -> cria rodada nova sem -RunDir
    select   05  regra pré-registrada -> finalists.json    (CPU)
    frozen   06  braço A: NF head com entrada mascarada    (GPU, treino real)
    retrain  07  braço B: SEDifferNet + NF head            (GPU, treino longo)
    report   08  métricas val/holdout + bootstrap          (GPU, inferência no teste)
    image    09  nível de imagem: SEDifferNet com fundo × sem fundo (final_models), pareado (GPU, inferência)
    all          convert -> train -> masks
    ablation     audit -> select -> frozen -> report

.EXAMPLE
  .\ablation_bg_removal\run_pipeline.ps1 -Stage all -Classes vari-grip -DryRun
  .\ablation_bg_removal\run_pipeline.ps1 -Stage ablation -Classes vari-grip -DryRun
  .\ablation_bg_removal\run_pipeline.ps1 -Stage audit -Classes vari-grip
  .\ablation_bg_removal\run_pipeline.ps1 -Stage frozen -Classes vari-grip -RunDir latest
#>
param(
    [ValidateSet('convert', 'train', 'masks', 'audit', 'select', 'frozen', 'retrain', 'report', 'image', 'all', 'ablation')]
    [string]$Stage = 'all',
    [string]$Classes = 'all',
    [string]$Model = 'yolo26s-seg.pt',
    [int]$Epochs = 100,
    [int]$Batch = 8,
    [double]$Conf = 0.10,
    [ValidateSet('yolo', 'nobg')]
    [string]$MaskSource = 'yolo',
    [string]$RunDir = '',
    [string]$Variants = '',
    [string]$Python = 'C:\Users\teo-s\.conda\envs\anomaly_attentideffernet\python.exe',
    [int]$Limit = 0,
    [switch]$Cross,
    [switch]$SkipExisting,
    [switch]$Overwrite,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$common = @('--classes', $Classes)
if ($Limit -gt 0) { $limitArgs = @('--limit', "$Limit") } else { $limitArgs = @() }
if ($DryRun) { $dry = @('--dry_run') } else { $dry = @() }
if ($SkipExisting) { $skip = @('--skip_existing') } else { $skip = @() }

function Invoke-Step([string]$Script, [string[]]$StepArgs) {
    Write-Host "`n>>> $Script $($StepArgs -join ' ')" -ForegroundColor Cyan
    # Avisos nativos no stderr (ex.: libjpeg "Corrupt JPEG data") não podem abortar o PS 5.1.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Python (Join-Path $here $Script) @StepArgs 2>&1 | ForEach-Object { "$_" }
    $code = $LASTEXITCODE
    $ErrorActionPreference = $prev
    if ($code -ne 0) { throw "$Script falhou (exit $code)" }
}

function RunArgs { if ($RunDir) { return @('--run_dir', $RunDir) } else { return @('--run_dir', 'latest') } }

if ($Stage -in @('convert', 'all')) {
    $a = $common + $limitArgs + $dry
    if ($Overwrite) { $a += '--overwrite' }
    Invoke-Step '01_build_yolo_dataset.py' $a
}
if ($Stage -in @('train', 'all')) {
    Invoke-Step '02_train_yolo_seg.py' ($common + @('--model', $Model, '--epochs', "$Epochs", '--batch', "$Batch") + $dry)
}
if ($Stage -in @('masks', 'all')) {
    if ($DryRun -and $Stage -eq 'all' -and $MaskSource -eq 'yolo') {
        Write-Host "`n>>> 03_generate_masks.py: dry-run omitido (depende dos pesos da etapa train)." -ForegroundColor Yellow
    } else {
        Invoke-Step '03_generate_masks.py' ($common + @('--source', $MaskSource, '--conf', "$Conf") + $limitArgs + $dry)
    }
}

if ($Stage -in @('audit', 'ablation')) {
    $a = $common + @('--mask_source', $MaskSource) + $limitArgs + $dry
    if ($RunDir) { $a += @('--run_dir', $RunDir) }
    Invoke-Step '04_audit_masks.py' $a
    if ($DryRun -and $Stage -eq 'ablation') {
        Write-Host "`n>>> Etapas 05-08: dry-run exige uma rodada com audit; rode -Stage audit e depois -Stage frozen -DryRun." -ForegroundColor Yellow
        return
    }
}
if ($Stage -in @('select', 'ablation')) {
    Invoke-Step '05_select_strategies.py' ($common + (RunArgs))
}
if ($Stage -in @('frozen', 'ablation')) {
    $a = $common + (RunArgs) + $skip + $dry
    if ($Variants) { $a += @('--variants', $Variants) }
    Invoke-Step '06_train_nf_head_masked.py' $a
}
if ($Stage -eq 'retrain') {
    $a = $common + (RunArgs) + $skip + $dry
    if ($Variants) { $a += @('--variants', $Variants) }
    Invoke-Step '07_retrain_encoder_masked.py' $a
}
if ($Stage -in @('report', 'ablation')) {
    Invoke-Step '08_report.py' ($common + (RunArgs) + $skip + $limitArgs + $dry)
}
if ($Stage -eq 'image') {
    $a = $common + $limitArgs + $dry
    if ($Cross) { $a += '--cross' }
    Invoke-Step '09_image_level_compare.py' $a
}
