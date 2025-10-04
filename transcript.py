#!/usr/bin/env python3
import argparse
from pathlib import Path

import kaldi_native_fbank as knf
import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import time
import os
print("Check if GPU is available:", end=" ")
print(torch.cuda.is_available())

def create_fbank():
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.remove_dc_offset = False
    opts.frame_opts.window_type = "hann"

    opts.mel_opts.low_freq = 0
    opts.mel_opts.num_bins = 128

    opts.mel_opts.is_librosa = True

    fbank = knf.OnlineFbank(opts)
    return fbank


def compute_features(audio, fbank):
    assert len(audio.shape) == 1, audio.shape
    fbank.accept_waveform(16000, audio)
    ans = []
    processed = 0
    while processed < fbank.num_frames_ready:
        ans.append(np.array(fbank.get_frame(processed)))
        processed += 1
    ans = np.stack(ans)
    return ans


def display(sess, model):
    print(f"=========={model} Input==========")
    for i in sess.get_inputs():
        print(i)
    print(f"=========={model }Output==========")
    for i in sess.get_outputs():
        print(i)


class OnnxModel:
    def __init__(
        self,
        encoder: str,
        decoder: str,
        joiner: str,
    ):
        self.init_encoder(encoder)
        display(self.encoder, "encoder")
        self.init_decoder(decoder)
        display(self.decoder, "decoder")
        self.init_joiner(joiner)
        display(self.joiner, "joiner")

    def init_encoder(self, encoder):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.encoder = ort.InferenceSession(
            encoder,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

        meta = self.encoder.get_modelmeta().custom_metadata_map
        self.normalize_type = meta["normalize_type"]
        print(meta)

        self.pred_rnn_layers = int(meta["pred_rnn_layers"])
        self.pred_hidden = int(meta["pred_hidden"])

    def init_decoder(self, decoder):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.decoder = ort.InferenceSession(
            decoder,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_joiner(self, joiner):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 1

        self.joiner = ort.InferenceSession(
            joiner,
            sess_options=session_opts,
            providers=["CPUExecutionProvider"],
        )

    def get_decoder_state(self):
        batch_size = 1
        state0 = torch.zeros(self.pred_rnn_layers, batch_size, self.pred_hidden).numpy()
        state1 = torch.zeros(self.pred_rnn_layers, batch_size, self.pred_hidden).numpy()
        return state0, state1

    def run_encoder(self, x: np.ndarray):
        # x: (T, C)
        x = torch.from_numpy(x)
        x = x.t().unsqueeze(0)
        # x: [1, C, T]
        x_lens = torch.tensor([x.shape[-1]], dtype=torch.int64)

        (encoder_out, out_len) = self.encoder.run(
            [
                self.encoder.get_outputs()[0].name,
                self.encoder.get_outputs()[1].name,
            ],
            {
                self.encoder.get_inputs()[0].name: x.numpy(),
                self.encoder.get_inputs()[1].name: x_lens.numpy(),
            },
        )
        # [batch_size, dim, T]
        return encoder_out

    def run_decoder(
        self,
        token: int,
        state0: np.ndarray,
        state1: np.ndarray,
    ):
        target = torch.tensor([[token]], dtype=torch.int32).numpy()
        target_len = torch.tensor([1], dtype=torch.int32).numpy()

        (decoder_out, decoder_out_length, state0_next, state1_next,) = self.decoder.run(
            [
                self.decoder.get_outputs()[0].name,
                self.decoder.get_outputs()[1].name,
                self.decoder.get_outputs()[2].name,
                self.decoder.get_outputs()[3].name,
            ],
            {
                self.decoder.get_inputs()[0].name: target,
                self.decoder.get_inputs()[1].name: target_len,
                self.decoder.get_inputs()[2].name: state0,
                self.decoder.get_inputs()[3].name: state1,
            },
        )
        return decoder_out, state0_next, state1_next

    def run_joiner(
        self,
        encoder_out: np.ndarray,
        decoder_out: np.ndarray,
    ):
        # encoder_out: [batch_size,  dim, 1]
        # decoder_out: [batch_size,  dim, 1]
        logit = self.joiner.run(
            [
                self.joiner.get_outputs()[0].name,
            ],
            {
                self.joiner.get_inputs()[0].name: encoder_out,
                self.joiner.get_inputs()[1].name: decoder_out,
            },
        )[0]
        # logit: [batch_size, 1, 1, vocab_size]
        return logit


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mmm format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def transcript_file_with_timestamps(encoder, decoder, joiner, tokens, wav_path) -> tuple:
    """
    Returns tuple of (text, timestamped_segments)
    where timestamped_segments is a list of (start_time, end_time, text) tuples
    """
    
    model = OnnxModel(encoder, decoder, joiner)

    id2token = dict()
    with open(tokens, encoding="utf-8") as f:
        for line in f:
            t, idx = line.split()
            id2token[int(idx)] = t

    start = time.time()
    fbank = create_fbank()
    audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio[:, 0]  # only use the first channel
    if sample_rate != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000,
        )
        sample_rate = 16000

    # Store original audio duration before padding
    original_audio_duration = audio.shape[0] / 16000
    
    tail_padding = np.zeros(sample_rate * 2)
    audio = np.concatenate([audio, tail_padding])

    blank = len(id2token) - 1
    ans = [blank]
    state0, state1 = model.get_decoder_state()
    decoder_out, state0_next, state1_next = model.run_decoder(ans[-1], state0, state1)

    features = compute_features(audio, fbank)
    if model.normalize_type != "":
        assert model.normalize_type == "per_feature", model.normalize_type
        features = torch.from_numpy(features)
        mean = features.mean(dim=1, keepdims=True)
        stddev = features.std(dim=1, keepdims=True) + 1e-5
        features = (features - mean) / stddev
        features = features.numpy()
    
    print(audio.shape)
    print("features.shape", features.shape)

    encoder_out = model.run_encoder(features)
    
    # Calculate actual frame duration based on original audio length (excluding padding)
    # The encoder processes the entire audio including padding, but we want timestamps 
    # relative to the original audio duration
    num_encoder_frames = encoder_out.shape[2]
    
    # Use original audio duration for more accurate timing
    frame_duration = original_audio_duration / num_encoder_frames
    
    print(f"Original audio duration: {original_audio_duration:.3f} seconds")
    print(f"Audio with padding: {(audio.shape[0] / 16000):.3f} seconds") 
    print(f"Number of encoder frames: {num_encoder_frames}")
    print(f"Frame duration: {frame_duration:.6f} seconds ({frame_duration*1000:.2f} ms per frame)")
    
    # Alternative: Calculate based on typical ASR frame rates
    # Many ASR models use 40ms frames with 10ms shift, so let's also try that
    typical_frame_shift = 0.01  # 10ms is common
    print(f"Expected frames for 10ms shift: {int(original_audio_duration / typical_frame_shift)}")
    
    # Use the frame duration that makes more sense
    if num_encoder_frames > 0:
        calculated_frame_duration = original_audio_duration / num_encoder_frames
        # If the calculated duration seems reasonable (between 5ms and 100ms per frame), use it
        if 0.005 <= calculated_frame_duration <= 0.1:
            frame_duration = calculated_frame_duration
        else:
            # Fall back to typical 10ms shift
            frame_duration = typical_frame_shift
            print(f"Using fallback frame duration: {frame_duration*1000:.2f} ms per frame")
    
    # For tracking timestamps
    timestamped_tokens = []  # List of (token_idx, frame_time) tuples
    current_word_start = None
    current_word_tokens = []
    
    # encoder_out:[batch_size, dim, T)
    for t in range(encoder_out.shape[2]):
        frame_time = t * frame_duration
        encoder_out_t = encoder_out[:, :, t : t + 1]
        logits = model.run_joiner(encoder_out_t, decoder_out)
        logits = torch.from_numpy(logits)
        logits = logits.squeeze()
        idx = torch.argmax(logits, dim=-1).item()
        
        if idx != blank:
            ans.append(idx)
            timestamped_tokens.append((idx, frame_time))
            state0 = state0_next
            state1 = state1_next
            decoder_out, state0_next, state1_next = model.run_decoder(
                ans[-1], state0, state1
            )

    end = time.time()

    elapsed_seconds = end - start
    audio_duration = audio.shape[0] / 16000
    real_time_factor = elapsed_seconds / audio_duration

    ans = ans[1:]  # remove the first blank
    tokens_list = [id2token[i] for i in ans]
    underline = "▁"
    text = "".join(tokens_list).replace(underline, " ").strip()

    # Create timestamped segments
    timestamped_segments = []
    if timestamped_tokens:
        current_word = ""
        word_start_time = timestamped_tokens[0][1]
        
        for i, (token_idx, frame_time) in enumerate(timestamped_tokens):
            token_text = id2token[token_idx]
            
            # Check if this token starts a new word (contains underline prefix)
            if token_text.startswith(underline):
                # Finish previous word if exists
                if current_word.strip():
                    word_end_time = timestamped_tokens[i-1][1] if i > 0 else frame_time
                    timestamped_segments.append((word_start_time, word_end_time, current_word.strip()))
                
                # Start new word
                current_word = token_text.replace(underline, " ")
                word_start_time = frame_time
            else:
                # Continue current word
                current_word += token_text
        
        # Add the last word
        if current_word.strip():
            word_end_time = timestamped_tokens[-1][1]
            timestamped_segments.append((word_start_time, word_end_time, current_word.strip()))

    print(ans)
    print(wav_path)
    print(text)
    print(f"RTF: {real_time_factor}")
    
    # Print timestamped results
    print("\n=== TIMESTAMPED TRANSCRIPTION ===")
    print(f"Total segments: {len(timestamped_segments)}")
    if timestamped_segments:
        total_duration = timestamped_segments[-1][1] - timestamped_segments[0][0]
        coverage_percent = (total_duration / original_audio_duration) * 100
        print(f"Transcribed duration: {total_duration:.3f} seconds")
        print(f"Audio coverage: {coverage_percent:.1f}% of {original_audio_duration:.3f}s total")
        
        # If coverage is low, it might indicate silence/padding at the end
        if coverage_percent < 80:
            print("Note: Low coverage may indicate silence at the end of the audio file")
    
    for start_time, end_time, word in timestamped_segments:
        print(f"[{format_timestamp(start_time)} --> {format_timestamp(end_time)}] {word}")
    
    return text, timestamped_segments


def transcript_file(encoder, decoder, joiner, tokens, wav_path) -> str:
    """
    Original function - returns only text for backward compatibility
    """
    text, segments = transcript_file_with_timestamps(encoder, decoder, joiner, tokens, wav_path)
    return text, segments


def save_timestamped_transcript(segments, output_file):
    """
    Save timestamped transcript to various formats
    """
    if output_file.endswith('.srt'):
        # SRT subtitle format
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, (start_time, end_time, text) in enumerate(segments, 1):
                # Convert to SRT time format (HH:MM:SS,mmm)
                start_srt = format_timestamp(start_time).replace('.', ',')
                end_srt = format_timestamp(end_time).replace('.', ',')
                f.write(f"{i}\n")
                f.write(f"{start_srt} --> {end_srt}\n")
                f.write(f"{text}\n\n")
    
    elif output_file.endswith('.vtt'):
        # WebVTT format
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("WEBVTT\n\n")
            for start_time, end_time, text in segments:
                start_vtt = format_timestamp(start_time)
                end_vtt = format_timestamp(end_time)
                f.write(f"{start_vtt} --> {end_vtt}\n")
                f.write(f"{text}\n\n")
    
    else:
        # Plain text with timestamps
        with open(output_file, 'w', encoding='utf-8') as f:
            for start_time, end_time, text in segments:
                f.write(f"[{format_timestamp(start_time)} --> {format_timestamp(end_time)}] {text}\n")


# ----------------------Example usage with timestamps ------------------------
# if __name__ == "__main__":
#     # Example usage with timestamps
#     text, segments = transcript_file_with_timestamps(
#         encoder=os.path.join("model_components","encoder.int8.onnx"),
#         decoder=os.path.join("model_components", "decoder.int8.onnx"),
#         joiner=os.path.join("model_components","joiner.int8.onnx"),
#         tokens=os.path.join("model_components","tokens.txt"),
#         wav_path="file.wav",
#     )
    
#     # Save timestamped transcript in different formats
#     save_timestamped_transcript(segments, "transcript.txt")
#     save_timestamped_transcript(segments, "transcript.srt")
#     save_timestamped_transcript(segments, "transcript.vtt")
    
#     print(f"\nFull text: {text}")
#     print(f"Generated {len(segments)} timestamped segments")
