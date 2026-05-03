param(
    [switch]$SkipPythonDeps,
    [switch]$SkipPiper,
    [switch]$SkipVoices,
    [switch]$InstallGpuDeps
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$src = Join-Path $root "src"
$packages = Join-Path $root ".packages"
$opusPackages = Join-Path $root ".packages-opus"
$pipCache = Join-Path $root ".pip-cache"
$tmpRoot = Join-Path $root ".tmp"
$modelCache = Join-Path $root ".model-cache"
$piperRoot = Join-Path $root ".piper-runtime"
$piperZip = Join-Path $piperRoot "piper_windows_amd64.zip"
$piperExe = Join-Path $piperRoot "piper\\piper.exe"
$voicesDir = Join-Path $piperRoot "voices"

$pythonDeps = @(
    "sounddevice>=0.4.6",
    "webrtcvad-wheels>=2.0.14",
    "numpy>=1.26",
    "faster-whisper>=1.2.1"
)

$opusPythonDeps = @(
    "huggingface-hub>=1.13.0",
    "sentencepiece>=0.2.0"
)

$gpuPythonDeps = @(
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12"
)

$voiceDownloads = @(
    @{ File = "en_US-amy-medium.onnx"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx" },
    @{ File = "en_US-amy-medium.onnx.json"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json" },
    @{ File = "es_ES-davefx-medium.onnx"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx" },
    @{ File = "es_ES-davefx-medium.onnx.json"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/davefx/medium/es_ES-davefx-medium.onnx.json" },
    @{ File = "fr_FR-siwis-medium.onnx"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx" },
    @{ File = "fr_FR-siwis-medium.onnx.json"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/siwis/medium/fr_FR-siwis-medium.onnx.json" },
    @{ File = "de_DE-thorsten-high.onnx"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx" },
    @{ File = "de_DE-thorsten-high.onnx.json"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json" },
    @{ File = "it_IT-paola-medium.onnx"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/paola/medium/it_IT-paola-medium.onnx" },
    @{ File = "it_IT-paola-medium.onnx.json"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/it/it_IT/paola/medium/it_IT-paola-medium.onnx.json" },
    @{ File = "pt_PT-tugao-medium.onnx"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_PT/tug%C3%A3o/medium/pt_PT-tug%C3%A3o-medium.onnx" },
    @{ File = "pt_PT-tugao-medium.onnx.json"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_PT/tug%C3%A3o/medium/pt_PT-tug%C3%A3o-medium.onnx.json" },
    @{ File = "pt_BR-faber-medium.onnx"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx" },
    @{ File = "pt_BR-faber-medium.onnx.json"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_BR/faber/medium/pt_BR-faber-medium.onnx.json" },
    @{ File = "ru_RU-irina-medium.onnx"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx" },
    @{ File = "ru_RU-irina-medium.onnx.json"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium/ru_RU-irina-medium.onnx.json" },
    @{ File = "vi_VN-vais1000-medium.onnx"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx" },
    @{ File = "vi_VN-vais1000-medium.onnx.json"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json" },
    @{ File = "zh_CN-huayan-medium.onnx"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx" },
    @{ File = "zh_CN-huayan-medium.onnx.json"; Uri = "https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json" }
)

function Write-Step([string]$Message) {
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Download-FileIfMissing([string]$Uri, [string]$OutPath) {
    if (Test-Path -LiteralPath $OutPath) {
        return
    }
    Invoke-WebRequest -Uri $Uri -OutFile $OutPath
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "Python was not found on PATH. Install Python 3.11+ first, then run this script again."
}

New-Item -ItemType Directory -Force -Path $packages | Out-Null
New-Item -ItemType Directory -Force -Path $opusPackages | Out-Null
New-Item -ItemType Directory -Force -Path $pipCache | Out-Null
New-Item -ItemType Directory -Force -Path $tmpRoot | Out-Null
New-Item -ItemType Directory -Force -Path $modelCache | Out-Null
New-Item -ItemType Directory -Force -Path $piperRoot | Out-Null
New-Item -ItemType Directory -Force -Path $voicesDir | Out-Null

$env:PIP_CACHE_DIR = $pipCache
$env:TMP = $tmpRoot
$env:TEMP = $tmpRoot

if (-not $SkipPythonDeps) {
    Write-Step "Installing local Python dependencies into .packages"
    & python -m pip install --disable-pip-version-check --upgrade --target $packages $pythonDeps
    if ($LASTEXITCODE -ne 0) {
        throw "Python dependency install failed."
    }

    Write-Step "Installing local OPUS translation dependencies into .packages-opus"
    & python -m pip install --disable-pip-version-check --upgrade --target $opusPackages $opusPythonDeps
    if ($LASTEXITCODE -ne 0) {
        throw "OPUS translation dependency install failed."
    }
}
else {
    Write-Step "Skipping Python dependency install"
}

if ($InstallGpuDeps) {
    Write-Step "Installing optional GPU runtime dependencies into .packages"
    & python -m pip install --disable-pip-version-check --upgrade --target $packages $gpuPythonDeps
    if ($LASTEXITCODE -ne 0) {
        throw "GPU runtime dependency install failed."
    }
}

if (-not $SkipPiper) {
    if (-not (Test-Path -LiteralPath $piperExe)) {
        Write-Step "Downloading Piper runtime"
        Invoke-WebRequest `
            -Uri "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_windows_amd64.zip" `
            -OutFile $piperZip

        Write-Step "Extracting Piper runtime"
        Expand-Archive -LiteralPath $piperZip -DestinationPath $piperRoot -Force
        Remove-Item -LiteralPath $piperZip -Force
    }
    else {
        Write-Step "Piper runtime already exists"
    }
}
else {
    Write-Step "Skipping Piper runtime install"
}

if (-not $SkipVoices) {
    Write-Step "Downloading default Piper voices"
    foreach ($voice in $voiceDownloads) {
        $destination = Join-Path $voicesDir $voice.File
        Download-FileIfMissing -Uri $voice.Uri -OutPath $destination
    }
}
else {
    Write-Step "Skipping Piper voice downloads"
}

$expectedPythonArtifacts = @(
    (Join-Path $packages "sounddevice.py"),
    (Join-Path $packages "numpy"),
    (Join-Path $packages "faster_whisper")
)

$expectedOpusArtifacts = @(
    (Join-Path $opusPackages "huggingface_hub"),
    (Join-Path $opusPackages "sentencepiece")
)

$expectedVoiceArtifacts = $voiceDownloads | ForEach-Object {
    Join-Path $voicesDir $_.File
}

Write-Step "Verifying local dependency files"
foreach ($artifact in $expectedPythonArtifacts) {
    if (-not (Test-Path -LiteralPath $artifact)) {
        throw "Missing expected dependency artifact: $artifact"
    }
}

Write-Step "Verifying local OPUS translation dependency files"
foreach ($artifact in $expectedOpusArtifacts) {
    if (-not (Test-Path -LiteralPath $artifact)) {
        throw "Missing expected OPUS dependency artifact: $artifact"
    }
}

Write-Step "Verifying default voice files"
foreach ($artifact in $expectedVoiceArtifacts) {
    if (-not (Test-Path -LiteralPath $artifact)) {
        throw "Missing expected voice artifact: $artifact"
    }
}

$voiceCount = @(Get-ChildItem -LiteralPath $voicesDir -Filter *.onnx -ErrorAction SilentlyContinue).Count

Write-Step "Setup complete"
Write-Host "Local packages: $packages"
Write-Host "OPUS packages:  $opusPackages"
Write-Host "Piper runtime:  $piperRoot"
Write-Host "Voice files:    $voiceCount detected in .piper-runtime\\voices"

if ($voiceCount -eq 0) {
    Write-Host ""
    Write-Host "No Piper voices were found." -ForegroundColor Yellow
    Write-Host "Add .onnx and .onnx.json voice files into .piper-runtime\\voices before launching the app."
}

Write-Host ""
Write-Host "Next step: pythonw CrossComms.pyw" -ForegroundColor Green
if (-not $InstallGpuDeps) {
    Write-Host "Optional GPU STT support: .\\setup.ps1 -InstallGpuDeps" -ForegroundColor DarkGray
}
Write-Host "Local OPUS translation models download automatically the first time a language pair is used." -ForegroundColor DarkGray
