import torch
import subprocess
import librosa
import numpy as np
import soundfile as sf
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering

from pyannote.audio import Pipeline, Audio
from pyannote.core import Segment
from pyannote.audio import Model, Inference

try:
    from silero_vad import VADIterator, load_silero_vad
except ImportError:
    print("Error: Could not import silero_vad. Ensure silero-vad==6.0.0 is installed correctly.")
    print("Try running: pip install silero-vad==6.0.0")

import os
import math
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count, current_process
import gradio as gr
import tempfile
import shutil

# Import your existing modules
from transcript import transcript_file
from utils import convert_to_wav, group_and_write_speakers

# Global variables
PIPELINE = None
EMBEDDING_MODEL = None
EMBEDING_INFERENCE = None

# Configuration constants
ENCODER = "model_components/encoder.int8.onnx"
DECODER = "model_components/decoder.int8.onnx"
JOINER = "model_components/joiner.int8.onnx"
TOKENS = "model_components/tokens.txt"
CHUNK_DURATION = 120
SAMPLE_RATE = 16000
VAD_THRESHOLD = 0.3
MIN_SPEECH_DURATION = 0.2
CLUSTER_DISTANCE_THRESHOLD = 0.7045654963945799
VAD_CHUNK_DURATION = 0.032

print("Check if GPU is available for transcription:", torch.cuda.is_available())

def load_pyannote_pipeline_diarize():
    """Load and return a pyannote pipeline."""
    config_file_path = os.path.join("model_components", "pyannote", "config_diarize.yaml")
    pipeline = Pipeline.from_pretrained(config_file_path, use_auth_token=False)
    return pipeline

def init_worker():
    """Initialize a pyannote pipeline and embedding model in each worker process."""
    global PIPELINE, EMBEDDING_MODEL, EMBEDING_INFERENCE
    print(f"Initializing pipeline in worker process: {current_process().name}")
    PIPELINE = load_pyannote_pipeline_diarize()
    EMBEDDING_MODEL = Model.from_pretrained("model_components/pyannote/wespeaker-voxceleb-resnet34-LM.bin")
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

def split_audio_ffmpeg(audio_file, start, end, out_file):
    cmd = [
        "ffmpeg", "-y", "-i", audio_file,
        "-ss", str(start), "-to", str(end),
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
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
    return True

def merge_same_speaker_segments(segments):
    """Merges consecutive audio segments if they have the same speaker."""
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
    """Perform speaker diarization using the pyannote pipeline."""
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

def process_chunk(args):
    """Process a single chunk of audio."""
    chunk_idx, audio_file, chunk_duration, num_speakers = args
    start_time = chunk_idx * chunk_duration
    chunk_file = f"chunk_{chunk_idx:03d}.wav"
    total_duration = get_audio_duration(audio_file)
    end_time = min(start_time + chunk_duration, total_duration)
    
    print(f"Splitting chunk {chunk_idx}: {start_time:.3f}s to {end_time:.3f}s")
    if not split_audio_ffmpeg(audio_file, start_time, end_time, chunk_file):
        return []
    
    print(f"Created {chunk_file}, size: {os.path.getsize(chunk_file)} bytes")
    segments = diarize_pyannote(chunk_file, num_speakers)
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
            embedding = EMBEDING_INFERENCE(seg_file)
        except Exception as e:
            print(f"Error extracting embedding for {seg_file}: {e}")
            embedding = np.zeros(256)  # Adjust size based on model output dim, wespeaker resnet34 typically 256
        
        text = transcribe_segment(seg_file)
        results.append((global_start, global_end, speaker, text, embedding))
        
        try:
            os.remove(seg_file)
        except Exception as e:
            print(f"Error removing {seg_file}: {e}")
    
    try:
        os.remove(chunk_file)
    except Exception as e:
        print(f"Error removing {chunk_file}: {e}")
    
    return results

def assign_global_speakers(all_results):
    """Assign global speaker labels using agglomerative clustering with improved logic."""
    if not all_results:
        return [], {}
    
    # Filter valid embeddings
    valid_results = [(start, end, local_speaker, text, emb) for start, end, local_speaker, text, emb in all_results 
                     if emb is not None and not np.all(emb == 0)]
    
    if not valid_results:
        print("No valid embeddings found for clustering.")
        return [], {}
    
    print(f"Total segments for clustering: {len(valid_results)}")
    
    # Debug: Show local speaker distribution
    local_speakers = {}
    for start, end, local_speaker, text, emb in valid_results:
        if local_speaker not in local_speakers:
            local_speakers[local_speaker] = 0
        local_speakers[local_speaker] += 1
    
    print(f"Local speaker distribution: {local_speakers}")
    
    embeddings = np.array([emb for _, _, _, _, emb in valid_results])
    
    # Normalize embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Calculate pairwise distances for debugging
    print(f"Embedding shape: {embeddings.shape}")
    distances = cdist(embeddings, embeddings, metric='cosine')
    print(f"Average cosine distance: {np.mean(distances):.4f}")
    print(f"Min cosine distance: {np.min(distances[distances > 0]):.4f}")
    print(f"Max cosine distance: {np.max(distances):.4f}")
    
    # More conservative cluster range - limit to reasonable number of speakers
    num_segments = len(embeddings)
    max_clusters = min(len(local_speakers) * 2, 8)  # At most 8 speakers, or 2x local speakers
    min_clusters = max(2, len(local_speakers) // 2)  # At least 2, but consider local speaker count
    
    best_score = -1
    best_n = min_clusters
    best_labels = None
    
    print(f"Testing cluster numbers from {min_clusters} to {max_clusters}...")
    
    # Try different clustering parameters
    for n in range(min_clusters, max_clusters + 1):
        try:
            # Use ward linkage for better clustering, fallback to average if ward fails
            try:
                clustering = AgglomerativeClustering(n_clusters=n, linkage='ward')
                # Ward requires Euclidean distance, so don't use cosine metric
                labels = clustering.fit_predict(embeddings)
            except ValueError:
                # Fallback to average linkage with cosine metric
                clustering = AgglomerativeClustering(n_clusters=n, metric='cosine', linkage='average')
                labels = clustering.fit_predict(embeddings)
            
            if len(np.unique(labels)) < 2:
                continue  # Skip if not enough clusters
            
            # Use cosine metric for silhouette score
            score = silhouette_score(embeddings, labels, metric='cosine')
            print(f"  - n_clusters={n}: silhouette_score={score:.4f}, unique_labels={len(np.unique(labels))}")
            
            # Prefer solutions with reasonable number of clusters and good score
            if score > best_score and len(np.unique(labels)) <= len(local_speakers) + 1:
                best_score = score
                best_n = n
                best_labels = labels
        except Exception as e:
            print(f"  - n_clusters={n}: failed with error {e}")
            continue
    
    if best_labels is None:
        print("Failed to find optimal clustering; using simple mapping based on local speakers.")
        # Fallback: create mapping based on local speaker labels
        unique_local_speakers = list(set([local_speaker for _, _, local_speaker, _, _ in valid_results]))
        speaker_mapping = {local_speaker: i for i, local_speaker in enumerate(unique_local_speakers)}
        best_labels = [speaker_mapping[local_speaker] for _, _, local_speaker, _, _ in valid_results]
        best_n = len(unique_local_speakers)
        best_score = 0.0
    
    print(f"Optimal number of clusters: {best_n} with score {best_score:.4f}")
    
    # Assign global speakers and collect mappings
    global_results = []
    speaker_embeddings = {}
    local_to_global_mapping = {}
    
    print("\nLocal to Global Speaker Mappings:")
    for i, (start, end, local_speaker, text, emb) in enumerate(valid_results):
        global_id = best_labels[i]
        speaker = f"SPEAKER_{global_id:02d}"
        global_results.append((start, end, speaker, text))
        
        if global_id not in speaker_embeddings:
            speaker_embeddings[global_id] = []
        speaker_embeddings[global_id].append(emb)
        
        # Track local to global mapping
        if local_speaker not in local_to_global_mapping:
            local_to_global_mapping[local_speaker] = {}
        if speaker not in local_to_global_mapping[local_speaker]:
            local_to_global_mapping[local_speaker][speaker] = 0
        local_to_global_mapping[local_speaker][speaker] += 1
        
        # Print the mapping for this segment
        print(f"Segment at {start:.3f}-{end:.3f}: Local {local_speaker} -> Global {speaker}")
    
    # Show mapping summary
    print("\nMapping Summary:")
    for local_speaker, global_assignments in local_to_global_mapping.items():
        most_common_global = max(global_assignments.items(), key=lambda x: x[1])
        total_segments = sum(global_assignments.values())
        print(f"Local {local_speaker}: {dict(global_assignments)} -> Most common: {most_common_global[0]} ({most_common_global[1]}/{total_segments} segments)")
    
    # Sort by start time
    global_results.sort(key=lambda x: x[0])
    
    return global_results, speaker_embeddings

def merge_global_segments(global_results):
    """Merges consecutive segments across the entire audio if they have the same global speaker."""
    if not global_results:
        return []
    merged = []
    current_start, current_end, current_speaker, current_text = global_results[0]
    for i in range(1, len(global_results)):
        next_start, next_end, next_speaker, next_text = global_results[i]
        if next_speaker == current_speaker and abs(current_end - next_start) < 1e-3:
            current_end = next_end
            current_text += " " + next_text
        else:
            merged.append((current_start, current_end, current_speaker, current_text))
            current_start = next_start
            current_end = next_end
            current_speaker = next_speaker
            current_text = next_text
    merged.append((current_start, current_end, current_speaker, current_text))
    return merged

def format_results_for_display(merged_results):
    """Format results for display in the text box."""
    formatted_text = "=== Transcription Results ===\n\n"
    for start, end, speaker, text in merged_results:
        formatted_text += f"[{start:.3f} - {end:.3f}] {speaker}: {text}\n\n"
    return formatted_text

def create_download_file(merged_results):
    """Create a downloadable text file with results."""
    now = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    temp_dir = tempfile.mkdtemp()
    filename = f"transcription_results_{now}.txt"
    filepath = os.path.join(temp_dir, filename)
    
    # Use your existing function to write the file
    group_and_write_speakers(merged_results, filepath, ignore_timestamps=False)
    
    return filepath

def initialize_models(progress=None):
    """Initialize models with progress updates."""
    global PIPELINE, EMBEDDING_MODEL, EMBEDING_INFERENCE
    
    try:
        if progress:
            progress(0.1, desc="Loading pyannote pipeline...")
        
        # Load pyannote pipeline
        config_file_path = os.path.join("model_components", "pyannote", "config_diarize.yaml")
        if not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Config file not found: {config_file_path}")
        
        PIPELINE = Pipeline.from_pretrained(config_file_path, use_auth_token=False)
        
        if progress:
            progress(0.2, desc="Loading embedding model...")
        
        # Load embedding model
        model_path = "model_components/pyannote/wespeaker-voxceleb-resnet34-LM.bin"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        EMBEDDING_MODEL = Model.from_pretrained(model_path)
        EMBEDING_INFERENCE = Inference(EMBEDDING_MODEL, window="whole")
        
        print("Models initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

def process_audio_file(audio_file, num_speakers, progress=gr.Progress()):
    """Main processing function for the Gradio interface."""
    if audio_file is None:
        return "Please upload an audio file.", None
    
    try:
        progress(0, desc="Starting transcription...")
        
        # Check if required model files exist
        required_files = [
            "model_components/pyannote/config_diarize.yaml",
            "model_components/pyannote/wespeaker-voxceleb-resnet34-LM.bin",
            ENCODER, DECODER, JOINER, TOKENS
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                return f"Required file not found: {file_path}\nPlease check your model_components directory.", None
        
        # Initialize the pipeline if not already done
        global PIPELINE, EMBEDDING_MODEL, EMBEDING_INFERENCE
        if PIPELINE is None or EMBEDDING_MODEL is None:
            progress(0.05, desc="Initializing models (this may take a few minutes)...")
            success = initialize_models(progress)
            if not success:
                return "Failed to initialize models. Please check the console for error messages.", None
        
        progress(0.3, desc="Processing audio file...")
        
        # Convert to wav if necessary
        audio_path = audio_file
        if not audio_file.lower().endswith(".wav"):
            progress(0.35, desc="Converting to WAV format...")
            try:
                audio_path = convert_to_wav(audio_file)
            except Exception as e:
                return f"Error converting audio file: {str(e)}", None
        
        # Get audio duration and split into chunks
        total_duration = get_audio_duration(audio_path)
        if total_duration == 0.0:
            return "Failed to get audio duration. Please check the audio file format and ensure ffmpeg is installed.", None
        
        progress(0.4, desc=f"Processing {total_duration:.1f}s audio...")
        num_chunks = math.ceil(total_duration / CHUNK_DURATION)
        print(f"Total duration: {total_duration:.3f}s, splitting into {num_chunks} chunks")
        
        # Process chunks
        all_results = []
        for chunk_idx in range(num_chunks):
            chunk_progress = 0.4 + (0.5 * chunk_idx / num_chunks)
            progress(chunk_progress, desc=f"Processing chunk {chunk_idx+1}/{num_chunks}")
            
            try:
                chunk_results = process_chunk((chunk_idx, audio_path, CHUNK_DURATION, num_speakers))
                all_results.extend(chunk_results)
                print(f"Chunk {chunk_idx+1} processed: {len(chunk_results)} segments")
            except Exception as e:
                print(f"Error processing chunk {chunk_idx}: {e}")
                continue
        
        progress(0.9, desc="Assigning global speakers...")
        all_results.sort(key=lambda x: x[0])
        
        if all_results:
            print(f"\n=== Speaker Assignment Phase ===")
            global_results, speaker_embeddings = assign_global_speakers(all_results)
            
            print(f"\n=== Merging Global Segments ===")
            merged_results = merge_global_segments(global_results)
            
            print(f"\n=== Combined Transcription (Global Speakers, Merged) ===")
            for start, end, speaker, text in merged_results:
                print(f"[{start:.3f} - {end:.3f}] {speaker}: {text}")
            
            # Format for display
            display_text = format_results_for_display(merged_results)
            
            # Create download file
            download_file = create_download_file(merged_results)
            
            progress(1.0, desc="Complete!")
            return display_text, download_file
        else:
            return "No transcription results found. Please check if the audio contains speech.", None
            
    except Exception as e:
        error_msg = f"Error during processing: {str(e)}\n\nPlease check:\n1. All model files are present\n2. FFmpeg is installed\n3. Audio file is valid"
        print(error_msg)
        return error_msg, None

# Create Gradio interface
def create_gradio_interface():
    with gr.Blocks(title="Audio Transcription & Diarization", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎵 Audio Transcription & Speaker Diarization")
        gr.Markdown("Upload an audio file to get transcribed text with speaker identification.")
        
        # Status indicator
        with gr.Row():
            status_text = gr.Textbox(
                label="System Status", 
                value="Ready to process audio files",
                interactive=False,
                lines=1
            )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input components
                audio_input = gr.File(
                    label="Upload Audio File",
                    file_types=[".mp3", ".wav", ".m4a", ".flac", ".ogg"],
                    type="filepath"
                )
                
                num_speakers = gr.Slider(
                    minimum=2,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Expected Number of Speakers"
                )
                
                process_btn = gr.Button("🚀 Start Transcription", variant="primary", size="lg")
                
                # Add model initialization button
                init_btn = gr.Button("🔧 Initialize Models", variant="secondary")
                
            with gr.Column(scale=2):
                # Output components
                output_text = gr.Textbox(
                    label="Transcription Results",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True,
                    placeholder="Transcription results will appear here..."
                )
                
                download_file = gr.File(
                    label="Download Results",
                    interactive=False
                )
        
        # Event handlers
        def init_models_ui():
            """Initialize models and update status."""
            try:
                success = initialize_models()
                if success:
                    return "✅ Models initialized successfully! Ready for transcription."
                else:
                    return "❌ Failed to initialize models. Check console for errors."
            except Exception as e:
                return f"❌ Error initializing models: {str(e)}"
        
        init_btn.click(
            fn=init_models_ui,
            inputs=[],
            outputs=[status_text]
        )
        
        process_btn.click(
            fn=process_audio_file,
            inputs=[audio_input, num_speakers],
            outputs=[output_text, download_file],
            show_progress=True
        )
        
        # Instructions and troubleshooting
        gr.Markdown("### 📝 Instructions:")
        gr.Markdown("""
        1. **Initialize Models** (optional): Click "Initialize Models" to pre-load models
        2. **Upload** your audio file (supports MP3, WAV, M4A, FLAC, OGG)
        3. **Set** the expected number of speakers
        4. **Click** "Start Transcription" and wait for processing
        5. **View** results in the text box above
        6. **Download** the results as a text file
        
        **Note:** First-time model loading may take several minutes.
        """)
        
        gr.Markdown("### 🔧 Troubleshooting:")
        gr.Markdown("""
        - **Stuck on initialization**: Check if all model files are present in `model_components/` directory
        - **FFmpeg errors**: Ensure FFmpeg is installed and accessible in PATH
        - **Out of memory**: Try processing shorter audio files or reduce chunk duration
        - **No results**: Check if audio file contains clear speech
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create a public link
        show_error=True
    )