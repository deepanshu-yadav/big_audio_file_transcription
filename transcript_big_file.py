import torch
import subprocess
import librosa
import numpy as np
import soundfile as sf
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cosine
from collections import Counter
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering  # Added import for clustering
from resemblyzer import VoiceEncoder, preprocess_wav
from scipy.stats import mode

from pyannote.audio import Pipeline, Audio
from pyannote.core import Segment
from pyannote.audio import Model, Inference
try:
    from silero_vad import load_silero_vad, get_speech_timestamps
except ImportError:
    print("Error: Could not import silero_vad. Please install it.")
    print("Try running: pip install silero-vad")
    exit(1)

import os
import math
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count, current_process

from transcript import transcript_file
from utils import convert_to_wav, group_and_write_speakers

# Global variables to hold the pipeline and embedding model, initialized per process
PIPELINE = None
EMBEDDING_MODEL = None
EMBEDING_INFERENCE = None
ENCODER_RESEMBLYZER = None

print("Check if GPU is available for transcription:", torch.cuda.is_available())

AUDIO_FILE = "output.wav"  # Change this to your input file path
ENCODER = "model_components/encoder.int8.onnx"
DECODER = "model_components/decoder.int8.onnx"
JOINER = "model_components/joiner.int8.onnx"
TOKENS = "model_components/tokens.txt"
CHUNK_DURATION = 120  # 2 minutes in seconds
NUM_SPEAKERS = 4
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.3  # Lowered for sensitivity
MIN_SPEECH_DURATION = 0.2  # Filter segments shorter than 0.2s
CLUSTER_DISTANCE_THRESHOLD = 0.7045654963945799
VAD_CHUNK_DURATION = 0.032  # 32ms = 512 samples at 16kHz

def load_pyannote_pipeline_diarize():
    """Load and return a pyannote pipeline."""
    config_file_path = os.path.join("model_components", "pyannote", "config_diarize.yaml")
    pipeline = Pipeline.from_pretrained(config_file_path, use_auth_token=False)
    return pipeline

def init_worker():
    """
    Initialize a pyannote pipeline and embedding model in each worker process.
    This function is called once at the start of each worker.
    """
    global PIPELINE, EMBEDDING_MODEL, EMBEDING_INFERENCE, ENCODER_RESEMBLYZER
    print(f"Initializing pipeline in worker process: {current_process().name}")
    PIPELINE = load_pyannote_pipeline_diarize()
    # Load model
    EMBEDDING_MODEL = Model.from_pretrained("model_components/pyannote/wespeaker-voxceleb-resnet34-LM.bin")
    ENCODER_RESEMBLYZER = VoiceEncoder() 
    # Initialize inference
    EMBEDING_INFERENCE = Inference(EMBEDDING_MODEL, window="whole")


def get_audio_duration(audio_file):
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", audio_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error in ffprobe: {result.stderr}")
        return 0.0
    return float(result.stdout)

def smooth_labels(labels, window_size=5):
    """
    Applies a majority vote smoothing to the labels.
    Corrects small, isolated labeling errors.
    """
    if window_size % 2 == 0:
        window_size += 1 # Ensure window size is odd
    
    smoothed_labels = np.copy(labels)
    half_window = window_size // 2
    
    for i in range(len(labels)):
        start = max(0, i - half_window)
        end = min(len(labels), i + half_window + 1)
        window = labels[start:end]
        
        # Find the most common label in the window
        most_common = mode(window, keepdims=True)[0][0]
        smoothed_labels[i] = most_common
        
    return smoothed_labels


def compute_affinity_matrix(embeddings):
    """Compute affinity matrix for spectral clustering based on cosine similarity."""
    n = len(embeddings)
    affinity = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            affinity[i, j] = sim
            affinity[j, i] = sim
    return affinity


def merge_diarization_output(diarization_result):
    """Merges consecutive segments from the same speaker."""
    if not diarization_result:
        return []

    merged_results = []
    current_start, current_end, current_speaker = diarization_result[0]

    for i in range(1, len(diarization_result)):
        next_start, next_end, next_speaker = diarization_result[i]
        
        if next_speaker == current_speaker:
            current_end = next_end
        else:
            merged_results.append((current_start, current_end, current_speaker))
            current_start, current_end, current_speaker = next_start, next_end, next_speaker
    
    merged_results.append((current_start, current_end, current_speaker))
    return merged_results


def diarize_lightweight(audio_file, num_speakers):
    """
    Performs speaker diarization using VAD, speaker embeddings, and spectral clustering.
    This version includes label smoothing for cleaner output.
    """
    print("Applying diarization...")
    # Load audio
    wav, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
    wav_tensor = torch.from_numpy(wav)
    
    # VAD
    vad_model = load_silero_vad()
    print("Running VAD to find speech regions...")
    speech_timestamps = get_speech_timestamps(
        wav_tensor,
        vad_model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,
        min_speech_duration_ms=500,
        min_silence_duration_ms=1000
    )

    if not speech_timestamps:
        print("Warning: No speech detected in the audio file.")
        return []

    print(f"Found {len(speech_timestamps)} major speech region(s).")

    # Windowing
    win_length_s = 1.5
    hop_length_s = 0.75
    win_length = int(win_length_s * SAMPLE_RATE)
    hop_length = int(hop_length_s * SAMPLE_RATE)
    
    embeddings = []
    window_times = []

    for region in speech_timestamps:
        region_start, region_end = region['start'], region['end']
        for start in range(region_start, region_end - win_length + 1, hop_length):
            end = start + win_length
            window_wav = wav[start:end]
            if np.std(window_wav) < 0.01: continue
            emb = ENCODER_RESEMBLYZER.embed_utterance(window_wav)
            embeddings.append(emb)
            window_times.append({'start': start / SAMPLE_RATE, 'end': end / SAMPLE_RATE})

    if not embeddings:
        print("Warning: Could not extract any valid embeddings.")
        return []

    print(f"Extracted {len(embeddings)} embeddings from overlapping windows.")
    
    embeddings_array = np.array(embeddings)
    
    # Clustering
    actual_num_speakers = min(num_speakers, len(embeddings_array))
    affinity_matrix = compute_affinity_matrix(embeddings_array)
    clustering = SpectralClustering(
        n_clusters=actual_num_speakers,
        affinity='precomputed',
        assign_labels='discretize',
        random_state=42
    )
    labels = clustering.fit_predict(affinity_matrix)
    print(f"Initial clustering: {Counter(labels)}")
    
    # *** NEW: Apply label smoothing ***
    labels = smooth_labels(labels, window_size=7)
    print(f"After smoothing:    {Counter(labels)}")
    
    # Merge results
    preliminary_results = []
    for i, label in enumerate(labels):
        start = window_times[i]['start']
        end = window_times[i]['end']
        preliminary_results.append((start, end, f"SPEAKER_{label:02d}"))
        
    final_results = merge_diarization_output(preliminary_results)
    
    return final_results

def split_audio_ffmpeg(audio_file, start, end, out_file):
    cmd = [
        "ffmpeg", "-y", "-i", audio_file,
        "-ss", str(start), "-to", str(end),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",  # Force mono
        "-c:a", "pcm_s16le",
        out_file
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        return False
    if not os.path.exists(out_file):
        print(f"Failed to create {out_file}")
        return False
    print(f"Created {out_file}, size: {os.path.getsize(out_file)} bytes")
    return True

def merge_same_speaker_segments(segments):
    """
    Merges consecutive audio segments if they have the same speaker.
    """
    if not segments:
        return []
    merged_segments = []
    current_start = segments[0][0]
    current_end = segments[0][1]
    current_speaker = segments[0][2]
    for i in range(1, len(segments)):
        next_start, next_end, next_speaker = segments[i]
        if next_speaker == current_speaker:
            current_end = next_end
        else:
            merged_segments.append((current_start, current_end, current_speaker))
            current_start = next_start
            current_end = next_end
            current_speaker = next_speaker
    merged_segments.append((current_start, current_end, current_speaker))
    return merged_segments


def diarize_pyannote(audio_file, max_speakers):
    """
    Perform speaker diarization using the pyannote pipeline.
    This function accesses the global PIPELINE variable which is
    initialized in the init_worker function for each process.
    """
    global PIPELINE
    if PIPELINE is None:
        raise RuntimeError("Pyannote pipeline is not initialized. Make sure init_worker is called.")
    diarization = PIPELINE(audio_file, max_speakers=max_speakers)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append((turn.start, turn.end, speaker))
    return segments


def transcribe_segment(segment_file):
    try:
        return transcript_file(
            wav_path=segment_file,
            encoder=ENCODER,
            decoder=DECODER,
            joiner=JOINER,
            tokens=TOKENS
        )
    except Exception as e:
        print(f"Error transcribing {segment_file}: {e}")
        return ""


def format_seconds(seconds: float) -> str:
    # Convert to integer and fractional seconds
    total_seconds = int(seconds)
    frac = seconds - total_seconds

    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    # Add fractional part back to seconds
    secs = secs + frac

    # Build formatted string
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    else:
        return f"{minutes}:{secs:05.2f}"


def get_embedding_from_file(file):

    wav, sr = librosa.load(file, sr=SAMPLE_RATE)
    emb = ENCODER_RESEMBLYZER.embed_utterance(wav)
    return emb


def process_chunk(chunk_idx, audio_file, chunk_duration, num_speakers,
                  method):
    """
    This function will be run by a worker process. It automatically
    uses the global PIPELINE, EMBEDDING_MODEL, and AUDIO variables that were initialized for it.
    """
    start_time = chunk_idx * chunk_duration
    chunk_file = f"chunk_{chunk_idx:03d}.wav"
    total_duration = get_audio_duration(audio_file)
    end_time = min(start_time + chunk_duration, total_duration)
    print(f"Splitting chunk {chunk_idx}: {start_time:.3f}s to {end_time:.3f}s")
    if not split_audio_ffmpeg(audio_file, start_time, end_time, chunk_file):
        return []
    if method=="pyannote":
        segments = diarize_pyannote(chunk_file, num_speakers)
    else:
        segments = diarize_lightweight(chunk_file, num_speakers)
    merged_segments = merge_same_speaker_segments(segments)
    results = []
    for idx, (start, end, speaker) in enumerate(merged_segments):
        seg_file = f"segment_{chunk_idx:03d}_{idx+1:03d}.wav"
        global_start = start + start_time
        global_end = end + start_time
        if not split_audio_ffmpeg(chunk_file, start, end, seg_file):
            continue
        print(f"Processing {seg_file} ({speaker}): {global_start:.3f} - {global_end:.3f}")
        # Extract embedding
        try:
            if method=="pyannote":
                embedding = EMBEDING_INFERENCE(seg_file)
            else:   
                embedding = get_embedding_from_file(seg_file)
        except Exception as e:
            print(f"Error extracting embedding for {seg_file}: {e}")
            embedding = np.zeros(256)  # Adjust size based on model output dim, wespeaker resnet34 typically 256
        text, timed_segments = transcribe_segment(seg_file)
        results.append((global_start, global_end, speaker, text, embedding, timed_segments))
        try:
            os.remove(seg_file)
        except Exception as e:
            print(f"Error removing {seg_file}: {e}")
    try:
        os.remove(chunk_file)
    except Exception as e:
        print(f"Error removing {chunk_file}: {e}")
    return results


def assign_global_speakers(all_results, num_speakers):
    """
    Assign global speaker labels using agglomerative clustering on embeddings,
    with optimal number of clusters determined by silhouette score.
    Prints mapping for each local speaker to global speaker.
    """
    if not all_results:
        return [], {}
    
    # Filter valid embeddings
    valid_results = [(start, end, local_speaker, text, emb, timed_segments) for start, end, local_speaker, text, emb, timed_segments  in all_results 
                     if emb is not None and not np.all(emb == 0) and not np.isnan(emb).any()]
    
    if not valid_results:
        print("No valid embeddings found for clustering.")
        return [], {}
    
    embeddings = np.array([emb for _, _, _, _, emb, _ in valid_results])
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Determine max possible clusters (e.g., min of 20 or number of segments / 2 to limit computation)
    num_segments = len(embeddings)
    max_clusters = min(20, num_segments // 2) if num_segments > 2 else 2
    
    best_score = -1
    best_n = 2
    best_labels = None
    
    print(f"Testing cluster numbers from 2 to {max_clusters}...")
    start_cluster = min(num_speakers, max_clusters+1)

    for n in range(start_cluster, max_clusters + 1):
        clustering = AgglomerativeClustering(n_clusters=n, metric='cosine', linkage='average')
        labels = clustering.fit_predict(embeddings)
        if len(np.unique(labels)) < 2:
            continue  # Skip if not enough clusters
        if len(embeddings) == 2 and len(labels) == 2:
            print("There are only two segments; and each of them belongs to a different speaker. skipping silhouette score calculation.")
            best_n = 2
            best_labels = labels
            break
        score = silhouette_score(embeddings, labels, metric='cosine')
        print(f"  - n_clusters={n}: silhouette_score={score:.4f}")
        if score > best_score:
            best_score = score
            best_n = n
            best_labels = labels
    
    if best_labels is None:
        print("Failed to find optimal clustering; falling back to single cluster.")
        best_labels = np.zeros(len(valid_results), dtype=int)
        best_n = 1
    
    print(f"Optimal number of clusters: {best_n} with score {best_score:.4f}")
    
    # Assign global speakers and collect mappings
    global_results = []
    speaker_embeddings = {}

    print("\nLocal to Global Speaker Mappings:")
    for i, (start, end, local_speaker, text, emb, timed_segments) in enumerate(valid_results):
        global_id = best_labels[i]
        speaker = f"SPEAKER_{global_id:02d}"
        global_results.append((start, end, speaker, text, timed_segments))
        if global_id not in speaker_embeddings:
            speaker_embeddings[global_id] = []
        speaker_embeddings[global_id].append(emb)
        # Print the mapping for this segment
        print(f"Segment at {start:.3f}-{end:.3f}: Local {local_speaker} -> Global {speaker}")
    
    # Sort by start time
    global_results.sort(key=lambda x: x[0])
    
    return global_results, speaker_embeddings


def merge_global_segments(global_results):
    """
    Merges consecutive segments across the entire audio if they have the same global speaker.
    """
    if not global_results:
        return []
    merged = []
    current_start, current_end, current_speaker, current_text, current_segments = global_results[0]
    for i in range(1, len(global_results)):
        next_start, next_end, next_speaker, next_text, next_segments = global_results[i]
        if next_speaker == current_speaker and abs(current_end - next_start) < 1e-3:
            current_end = next_end
            current_text += " " + next_text
            current_segments.extend(next_segments)
        else:
            merged.append((current_start, current_end, current_speaker, current_text, current_segments))
            current_start = next_start
            current_end = next_end
            current_speaker = next_speaker
            current_text = next_text
            current_segments = next_segments
    # Append the final segment
    merged.append((current_start, current_end, current_speaker, current_text, current_segments))
    return merged


def process_chunk_wrapper(args):
    return process_chunk(*args)


def transcript_file_with_diarization(audio_file,
                                     chunk_duration,
                                     tentative_speakers,
                                     method="resemblyzer"):
    # Convert to wav if not already wav
    
    if not audio_file.lower().endswith(".wav"):
        print(f"Converting {audio_file} to WAV...")
        audio_file = convert_to_wav(audio_file)
    total_duration = get_audio_duration(audio_file)
    start_time = time.time()
    if total_duration == 0.0:
        print("Failed to get audio duration. Check if audio_file exists and is valid.")
        return
    print(f"Total duration: {total_duration:.3f}s")
    num_chunks = math.ceil(total_duration / chunk_duration)
    print(f"Total duration: {total_duration:.3f}s, splitting into {num_chunks} chunks")
    # Use the initializer to set up the pipeline and embedding in each worker

    args = [ (ch, audio_file, chunk_duration, tentative_speakers, method) for ch in range(num_chunks)]
    with Pool(processes=cpu_count() // 4, initializer=init_worker) as pool:
        chunk_results = pool.map(process_chunk_wrapper, args)
    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
    all_results = []
    for results in chunk_results:
        all_results.extend(results)
    all_results.sort(key=lambda x: x[0])
    # Global speaker assignment with clustering
    # format all the timestamps to H:MM:SS.sss

    display_text = ""
    if all_results:
        global_results, speaker_embeddings = assign_global_speakers(
            all_results, tentative_speakers)
        # Merge consecutive global segments
        merged_results = merge_global_segments(global_results)
        # Store average embeddings in files (optional, commented out)
        # os.makedirs("speaker_embeddings", exist_ok=True)
        # for speaker_id, embs in speaker_embeddings.items():
        #     avg_emb = np.mean(embs, axis=0)
        #     speaker_label = f"SPEAKER_{speaker_id:02d}"
        #     np.save(os.path.join("speaker_embeddings", f"{speaker_label}_embedding.npy"), avg_emb)
        print("\n=== Combined Transcription (Global Speakers, Merged) ===")
        # format all the timestamps to H:MM:SS.sss
        merged_results = [ (format_seconds(s), format_seconds(e), sp, te) for s, e, sp, te, seg  in merged_results]
        for start, end, speaker, text in merged_results:
            line = f"[{start} - {end}] {speaker}: {text}"
            print(line)
            display_text += line + "\n"
        # Dump results to file
        # dump_transcription_results(merged_results)
        now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        filename = f"transcription_results_{now}.txt"
        group_and_write_speakers(merged_results, filename, ignore_timestamps=False)
        file_fullname = os.path.abspath(filename)
        print(f"\nTranscription results saved to {file_fullname}")
        return display_text, file_fullname
    else:
        print("No results to process.")

# Example usage:
# if __name__ == "__main__":
#     transcript_file_with_diarization(AUDIO_FILE, CHUNK_DURATION, NUM_SPEAKERS)