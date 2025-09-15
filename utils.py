import os
import subprocess
from datetime import datetime
from collections import defaultdict

def convert_to_wav(input_file, output_file=None, sample_rate=16000):
    """
    Converts any audio file (mp3, m4a, flac, etc.) to mono WAV format using ffmpeg.
    Returns the path to the WAV file.
    """
    if output_file is None:
        base, _ = os.path.splitext(input_file)
        output_file = base + ".wav"
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-ar", str(sample_rate),
        "-ac", "1",  # mono
        "-c:a", "pcm_s16le",
        output_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr}")
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Failed to create {output_file}")
    return output_file


def dump_transcription_results(results, output_dir="."):
    """
    Dumps the transcription results to a text file named
    transcription_results_YYYYMMDD_HHMMSS.txt in the specified directory.
    Each line: [start - end] speaker: text
    """
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcription_results_{now}.txt"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        for start, end, speaker, text in results:
            f.write(f"[{start:.3f} - {end:.3f}] {speaker}: {text}\n")
    print(f"Transcription results saved to {filepath}")
    return filepath


def group_and_write_speakers(global_results, output_file, ignore_timestamps=True):
    """
    Groups all text by speaker from global_results and writes to output_file.
    global_results: list of (start, end, speaker, text)
    ignore_timestamps: if True, only speaker and text are written.
    """
    

    speaker_texts = defaultdict(list)
    for start, end, speaker, text in global_results:
        if ignore_timestamps:
            speaker_texts[speaker].append(text)
        else:
            speaker_texts[speaker].append(f"[{start:.3f} - {end:.3f}] {text}")

    with open(output_file, "w", encoding="utf-8") as f:
        for speaker, texts in speaker_texts.items():
            f.write(f"{speaker}:\n")
            for t in texts:
                f.write(f"{t}\n")
            f.write("\n")