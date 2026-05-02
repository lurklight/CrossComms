param(
    [Parameter(Mandatory = $true)]
    [string]$AudioPath,

    [string]$Culture = "en-US"
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Speech

$engine = [System.Speech.Recognition.SpeechRecognitionEngine]::new()

$grammar = New-Object System.Speech.Recognition.DictationGrammar
$engine.LoadGrammar($grammar)
$engine.SetInputToWaveFile($AudioPath)

$result = $engine.Recognize()
if ($null -ne $result -and -not [string]::IsNullOrWhiteSpace($result.Text)) {
    Write-Output $result.Text.Trim()
}
