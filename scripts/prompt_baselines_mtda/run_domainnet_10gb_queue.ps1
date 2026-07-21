[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [string]$OutputRoot = "",
    [string]$PythonBin = "python.exe",
    [int]$Seed = 100,
    [int]$CoopTrainBatchSize = 16,
    [int]$CocoopTrainBatchSize = 2,
    [int]$TestBatchSize = 16,
    [int]$NumWorkers = 2,
    [ValidateSet("coop", "cocoop")]
    [string[]]$Methods = @("coop", "cocoop"),
    [ValidateSet("C", "I", "P", "Q", "R", "S")]
    [string[]]$Sources = @("C", "I", "P", "Q", "R", "S")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..\..")
)
$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot "output\domainnet_prompt_baselines"
}
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$env:PYTHONUTF8 = "1"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

Set-Location $RepoRoot

& $PythonBin scripts/datasets/verify_domainnet_layout.py --root $DataRoot
if ($LASTEXITCODE -ne 0) {
    throw "DomainNet validation failed; no training was started."
}

Write-Host "Low-memory DomainNet source-only queue"
Write-Host "  Data:       $DataRoot"
Write-Host "  Output:     $OutputRoot"
Write-Host "  Seed:       $Seed"
Write-Host "  CoOp batch: $CoopTrainBatchSize"
Write-Host "  CoCoOp:     $CocoopTrainBatchSize"
Write-Host "  Test batch: $TestBatchSize"

$SourceSpecs = [ordered]@{
    C = @{ Domain = "clipart";  Targets = @("infograph", "painting", "quickdraw", "real", "sketch") }
    I = @{ Domain = "infograph"; Targets = @("clipart", "painting", "quickdraw", "real", "sketch") }
    P = @{ Domain = "painting"; Targets = @("clipart", "infograph", "quickdraw", "real", "sketch") }
    Q = @{ Domain = "quickdraw"; Targets = @("clipart", "infograph", "painting", "real", "sketch") }
    R = @{ Domain = "real"; Targets = @("clipart", "infograph", "painting", "quickdraw", "sketch") }
    S = @{ Domain = "sketch"; Targets = @("clipart", "infograph", "painting", "quickdraw", "real") }
}

$MethodSpecs = @{
    coop = @{ Name = "coop"; Trainer = "CoOpMTDA"; Config = "configs/trainers/PromptBaselineMTDA/coop_vit_b16.yaml"; Batch = $CoopTrainBatchSize }
    cocoop = @{ Name = "cocoop"; Trainer = "CoCoOpMTDA"; Config = "configs/trainers/PromptBaselineMTDA/cocoop_vit_b16.yaml"; Batch = $CocoopTrainBatchSize }
}

foreach ($MethodName in $Methods) {
    $Method = $MethodSpecs[$MethodName]
    foreach ($Source in $Sources) {
        $Spec = $SourceSpecs[$Source]
        $RunDir = Join-Path $OutputRoot "$($Method.Name)\source_only\${Source}2O\seed$Seed"
        $Metrics = Join-Path $RunDir "mtda_metrics.json"
        if ((Test-Path -LiteralPath $Metrics) -and
            (Get-Item -LiteralPath $Metrics).Length -gt 0) {
            Write-Host "Skipping completed run: $($Method.Name) ${Source}2O"
            continue
        }

        Write-Host "[$(Get-Date -Format o)] START $($Method.Name) ${Source}2O"
        $TrainArgs = @(
            "train.py",
            "--root", $DataRoot,
            "--seed", "$Seed",
            "--trainer", $Method.Trainer,
            "--dataset-config-file", "configs/datasets/domainnet_mtda.yaml",
            "--config-file", $Method.Config,
            "--source-domains", $Spec.Domain,
            "--target-domains"
        ) + $Spec.Targets + @(
            "--output-dir", $RunDir,
            "TRAIN.MAX_BATCHES_PER_EPOCH", "-1",
            "TRAIN.SOURCE_ONLY", "True",
            "TRAINER.PROMPT_BASELINE_MTDA.MIX_TARGETS", "True",
            "TRAINER.PROMPT_BASELINE_MTDA.LAMBDA_ENT", "0.0",
            "DATALOADER.TRAIN_X.BATCH_SIZE", "$($Method.Batch)",
            "DATALOADER.TRAIN_U.SAME_AS_X", "True",
            "DATALOADER.TRAIN_U.BATCH_SIZE", "$($Method.Batch)",
            "DATALOADER.TEST.BATCH_SIZE", "$TestBatchSize",
            "DATALOADER.NUM_WORKERS", "$NumWorkers"
        )
        & $PythonBin @TrainArgs
        if ($LASTEXITCODE -ne 0) {
            throw "$($Method.Name) ${Source}2O failed with exit code $LASTEXITCODE"
        }

        & $PythonBin scripts/prompt_baselines_mtda/collect_results.py `
            --run-dir $RunDir `
            --method $Method.Name `
            --protocol source_only `
            --source $Source `
            --seed $Seed `
            --entropy-weight 0.0 `
            --train-batch-size $Method.Batch `
            --test-batch-size $TestBatchSize `
            --expected-targets 5
        if ($LASTEXITCODE -ne 0) {
            throw "Result collection failed for $($Method.Name) ${Source}2O"
        }
        Write-Host "[$(Get-Date -Format o)] DONE $($Method.Name) ${Source}2O"
    }
}

Write-Host "All requested CoOp and CoCoOp DomainNet runs are complete."
