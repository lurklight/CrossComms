# CrossComms

`CrossComms` is a Windows desktop app that listens to your mic, translates what you say, turns it back into speech, and sends that speech into a virtual microphone for Discord, in-game voice chat, or other apps.

## What It Does

- Captures your microphone live
- Transcribes your speech with local `faster-whisper`
- Translates the text
- Speaks the translated result with local Piper voices
- Routes the translated voice into a virtual mic such as `VB-CABLE`

## What You Need First

Before you run CrossComms, make sure you have:

- `Windows`
- `Python 3.11+` installed and available on `PATH`
- `VB-CABLE` installed if you want the translated voice to appear as a microphone in Discord or games
- a working microphone
- internet access for the translation step

Important:

- The app setup script installs the Python packages, Piper runtime, and default Piper voices for you
- The app does not install `Python` or `VB-CABLE` for you

## Setup Checklist

1. Install `Python 3.11+`
2. Install `VB-CABLE`
3. Open the folder where you cloned or downloaded `CrossComms`
4. Run:

```powershell
.\setup.ps1
```

Or just double-click:

```text
setup.bat
```

5. Launch the app with:

```powershell
pythonw CrossComms.pyw
```

Or double-click:

```text
CrossComms.pyw
```

## What Setup Downloads

The setup script downloads and installs:

- local Python dependencies into `.packages`
- Piper runtime into `.piper-runtime\piper`
- the default Piper voice set into `.piper-runtime\voices`

Default voices included by setup:

- English
- Spanish
- French
- German
- Italian
- Portuguese
- Portuguese (Brazil)
- Russian
- Vietnamese
- Chinese (Simplified)

## VB-CABLE Checklist

If you want CrossComms to speak into Discord or a game, use this routing:

1. In CrossComms:
   `Input Microphone` = your real mic
2. In CrossComms:
   `Virtual Cable Out` = `CABLE Input (VB-Audio Virtual Cable)`
3. In Discord or your game:
   microphone/input device = `CABLE Output (VB-Audio Virtual Cable)`

That means:

- your real mic goes into CrossComms
- CrossComms speaks into `CABLE Input`
- Discord/game listens to `CABLE Output`

## First App Test

After launch:

1. Pick your `Source Language`
2. Pick your `Target Language`
3. Pick your real mic as `Input Microphone`
4. Pick `CABLE Input` as `Virtual Cable Out`
5. Click `Start`
6. Say a short phrase

If you just want to test without talking, use the `Manual Test` section in the app.

## Supported Languages

Current built-in language list:

- `en` English
- `es` Spanish
- `fr` French
- `de` German
- `it` Italian
- `pt` Portuguese
- `pt-BR` Portuguese (Brazil)
- `ru` Russian
- `vi` Vietnamese
- `zh-CN` Chinese (Simplified)

These are defined in [languages.json](languages.json).

## Adding More Languages

To add another language:

1. Add it to [languages.json](languages.json)
2. Add the matching Piper `.onnx` and `.onnx.json` voice files to `.piper-runtime\voices`
3. If you want public setup to download that voice automatically too, add it to [setup.ps1](setup.ps1)
4. Restart the app

Example:

```json
{
  "code": "pl",
  "name": "Polish",
  "voice_family": "pl",
  "default_target_voice": "pl_PL-darkman-medium.onnx"
}
```

## Current Limits

This build works, but it is still an MVP.

Current limitations:

- translation still needs internet
- latency is not fully optimized yet
- it is not fully streaming end-to-end yet
- mic sensitivity is not exposed as a simple UI slider yet
- VB-CABLE routing still depends on correct Windows device selection

## Handy Commands

Run tests:

```powershell
python -m unittest discover -s tests
```

Compile check:

```powershell
python -m compileall src app.py CrossComms.pyw
```

Launch with console hidden:

```powershell
pythonw CrossComms.pyw
```
