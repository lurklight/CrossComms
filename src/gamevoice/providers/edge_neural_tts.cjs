const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const projectRoot = path.resolve(__dirname, "..", "..", "..");
const { EdgeTTS, Constants } = require(path.join(
  projectRoot,
  ".node-runtime",
  "node_modules",
  "@andresaya",
  "edge-tts"
));
const ffmpegPath = require(path.join(
  projectRoot,
  ".node-runtime",
  "node_modules",
  "ffmpeg-static"
));

function parseArgs(argv) {
  const parsed = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) {
      continue;
    }
    parsed[token.slice(2)] = argv[index + 1];
    index += 1;
  }
  return parsed;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const textPath = args["text-path"];
  const outputPath = args["output-path"];
  const voice = args["voice"] || "en-US-AriaNeural";
  const sampleRate = Number(args["sample-rate"] || "48000");

  if (!textPath || !outputPath) {
    throw new Error("Missing required --text-path or --output-path argument.");
  }

  const text = fs.readFileSync(textPath, "utf8").trim();
  if (!text) {
    throw new Error("No text was provided for neural TTS.");
  }

  const outputBase = outputPath.replace(/\.[^.]+$/, "");
  const mp3Path = `${outputBase}.mp3`;
  const tts = new EdgeTTS();

  try {
    await tts.synthesize(text, voice, {
      outputFormat: Constants.OUTPUT_FORMAT.AUDIO_24KHZ_96KBITRATE_MONO_MP3,
      rate: -8,
      volume: 95,
    });
    await tts.toFile(outputBase, "mp3");

    const ffmpegResult = spawnSync(
      ffmpegPath,
      [
        "-y",
        "-i",
        mp3Path,
        "-ac",
        "1",
        "-ar",
        String(sampleRate),
        "-sample_fmt",
        "s16",
        outputPath,
      ],
      {
        encoding: "utf8",
        windowsHide: true,
      }
    );
    if (ffmpegResult.status !== 0) {
      throw new Error(
        (ffmpegResult.stderr || ffmpegResult.stdout || "ffmpeg conversion failed").trim()
      );
    }
    if (!fs.existsSync(outputPath) || fs.statSync(outputPath).size === 0) {
      throw new Error("Neural TTS conversion produced an empty WAV file.");
    }
  } finally {
    if (fs.existsSync(mp3Path)) {
      fs.unlinkSync(mp3Path);
    }
  }
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
