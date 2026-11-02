# The source-plugin resolver provisions the node's dependencies automatically.
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $ConsumerArgs
)
$ErrorActionPreference = 'Stop'
$bundledClient = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../clients/python'))
if (-not $env:REMOTEMEDIA_PYTHON_SRC -and (Test-Path (Join-Path $bundledClient 'remotemedia'))) {
    $env:REMOTEMEDIA_PYTHON_SRC = $bundledClient
}
if ($env:REMOTEMEDIA_PYTHON_SRC) {
    $clientRoot = [IO.Path]::GetFullPath($env:REMOTEMEDIA_PYTHON_SRC)
    if (-not (Test-Path (Join-Path $clientRoot 'remotemedia'))) {
        throw 'REMOTEMEDIA_PYTHON_SRC must point to the SDK clients/python directory'
    }
    $sourcePaths = $clientRoot
    $repositoryRoot = [IO.Path]::GetFullPath((Join-Path $clientRoot '../..'))
    if (Test-Path (Join-Path $repositoryRoot 'remotemedia/protos')) {
        $sourcePaths += ';' + $repositoryRoot
    }
    $env:PYTHONPATH = if ($env:PYTHONPATH) { $sourcePaths + ';' + $env:PYTHONPATH } else { $sourcePaths }
}
if (-not $env:REMOTEMEDIA_NODE_TIMEOUT_MS) { $env:REMOTEMEDIA_NODE_TIMEOUT_MS = '600000' }
& python (Join-Path $PSScriptRoot 'consume.py') @ConsumerArgs
exit $LASTEXITCODE
