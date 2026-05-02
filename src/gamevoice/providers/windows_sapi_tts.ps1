param(
    [Parameter(Mandatory = $true)]
    [string]$TextPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [string]$Language = "en"
)

$ErrorActionPreference = "Stop"

$text = Get-Content -LiteralPath $TextPath -Raw -Encoding UTF8
if ([string]::IsNullOrWhiteSpace($text)) {
    throw "No text was provided for TTS."
}

$voice = New-Object -ComObject SAPI.SpVoice
$descriptions = @{
    en = @("english")
    es = @("spanish", "espanol")
    fr = @("french", "francais")
    de = @("german", "deutsch")
    ja = @("japanese", "nihongo")
    zh = @("chinese", "mandarin")
}

$tokens = $descriptions[$Language.ToLowerInvariant()]
if ($tokens) {
    $voices = $voice.GetVoices()
    for ($index = 0; $index -lt $voices.Count; $index++) {
        $candidate = $voices.Item($index)
        $description = $candidate.GetDescription().ToLowerInvariant()
        if ($tokens | Where-Object { $description.Contains($_) }) {
            $voice.Voice = $candidate
            break
        }
    }
}

$stream = New-Object -ComObject SAPI.SpFileStream
try {
    $stream.Open($OutputPath, 3, $false)
    $voice.AudioOutputStream = $stream
    $voice.Speak($text) | Out-Null
}
finally {
    if ($stream) {
        $stream.Close()
    }
}
