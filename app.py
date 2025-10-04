import gradio as gr
from transcript_big_file import transcript_file_with_diarization  

def transcribe_audio(audio_file, num_speakers, chunk_duration, diarization_method):
    """
    Wrapper function for Gradio to handle transcription with diarization.
    
    Args:
        audio_file (str): Path to the uploaded audio file.
        num_speakers (int): Number of tentative speakers.
        chunk_duration (float): Chunk duration in seconds.
        diarization_method (str): Selected diarization method.
    
    Returns:
        tuple: (transcription_text: str, output_file_path: str)
    """
    if audio_file is None:
        return "Please upload an audio file.", None
    
    # Pass the selected diarization method
    transcription_text, output_file_path = transcript_file_with_diarization(
        audio_file, chunk_duration, num_speakers, diarization_method
    )
    
    return transcription_text, output_file_path

# Create the Gradio interface
with gr.Blocks(title="Audio Transcription with Diarization") as demo:
    gr.Markdown("# Audio Transcription with Speaker Diarization")
    gr.Markdown("Upload an audio file, specify the number of tentative speakers, chunk duration, and diarization method.")
    
    with gr.Row():
        audio_input = gr.Audio(
            label="Upload Audio File",
            type="filepath"  # Specifies that the output is a file path
        )
    
    with gr.Row():
        num_speakers = gr.Number(
            label="Number of Tentative Speakers",
            value=2,
            minimum=1,
            step=1
        )
        chunk_duration = gr.Number(
            label="Chunk Duration (seconds)",
            value=120.0,
            minimum=1.0,
            step=1.0
        )
        diarization_method = gr.Dropdown(
            label="Diarization Method",
            choices=["resemblyzer", "pyannote"],
            value="resemblyzer"
        )
    
    transcribe_btn = gr.Button("Transcribe", variant="primary")
    
    with gr.Row():
        transcription_output = gr.Textbox(
            label="Transcription Results",
            lines=10,
            max_lines=20
        )
        download_output = gr.File(
            label="Download Transcription File"
        )
    
    # Event handler
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, num_speakers, chunk_duration, diarization_method],
        outputs=[transcription_output, download_output]
    )
    
    gr.Markdown("### Notes:")
    gr.Markdown("- The transcription includes speaker diarization based on the provided parameters.")
    gr.Markdown("- Download the file for the full formatted output.")

# Launch the app
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)