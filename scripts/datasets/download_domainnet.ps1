[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DataRoot,
    [string]$DownloadDir = "",
    [bool]$KeepArchives = $true
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$DataRoot = [System.IO.Path]::GetFullPath($DataRoot)
if ([string]::IsNullOrWhiteSpace($DownloadDir)) {
    $DownloadDir = Join-Path $DataRoot "downloads\domainnet"
}
$DownloadDir = [System.IO.Path]::GetFullPath($DownloadDir)
$TargetDir = Join-Path $DataRoot "DomainNet"
$ListDir = Join-Path $TargetDir "image_list"
$BaseUrl = if ($env:DOMAINNET_BASE_URL) {
    $env:DOMAINNET_BASE_URL.TrimEnd("/")
} else {
    "https://csr.bu.edu/ftp/visda/2019/multi-source"
}
$RepoRoot = [System.IO.Path]::GetFullPath(
    (Join-Path $PSScriptRoot "..\..")
)
$Domains = @("clipart", "infograph", "painting", "quickdraw", "real", "sketch")

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    throw "curl.exe is required (it is included with current Windows releases)."
}
if (-not (Get-Command python.exe -ErrorAction SilentlyContinue)) {
    throw "python.exe is not available in the active environment."
}

New-Item -ItemType Directory -Force -Path $DownloadDir, $ListDir | Out-Null

function Get-ResumableFile {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination
    )

    if ((Test-Path -LiteralPath $Destination) -and
        (Get-Item -LiteralPath $Destination).Length -gt 0) {
        Write-Host "Using existing file: $Destination"
        return
    }

    $Partial = "$Destination.part"
    Write-Host "Downloading: $Url"
    & curl.exe `
        --fail `
        --location `
        --retry 8 `
        --retry-all-errors `
        --continue-at - `
        --output $Partial `
        $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed with curl exit code $LASTEXITCODE`: $Url"
    }
    Move-Item -Force -LiteralPath $Partial -Destination $Destination
}

foreach ($Domain in $Domains) {
    $Archive = Join-Path $DownloadDir "$Domain.zip"
    if ($Domain -in @("clipart", "painting")) {
        $ArchiveUrl = "$BaseUrl/groundtruth/$Domain.zip"
    } else {
        $ArchiveUrl = "$BaseUrl/$Domain.zip"
    }
    Get-ResumableFile -Url $ArchiveUrl -Destination $Archive

    $DomainDir = Join-Path $TargetDir $Domain
    if (-not (Test-Path -LiteralPath $DomainDir -PathType Container)) {
        Write-Host "Extracting $Archive"
        Expand-Archive -LiteralPath $Archive -DestinationPath $TargetDir -Force
    } else {
        Write-Host "Domain directory exists; extraction skipped: $DomainDir"
    }

    foreach ($Split in @("train", "test")) {
        $ListName = "${Domain}_${Split}.txt"
        Get-ResumableFile `
            -Url "$BaseUrl/domainnet/txt/$ListName" `
            -Destination (Join-Path $ListDir $ListName)
    }

    if (-not $KeepArchives) {
        Remove-Item -Force -LiteralPath $Archive
    }
}

& python.exe `
    (Join-Path $RepoRoot "scripts\datasets\verify_domainnet_layout.py") `
    --root $DataRoot
if ($LASTEXITCODE -ne 0) {
    throw "DomainNet layout verification failed."
}

Write-Host "DomainNet is ready under $TargetDir"
