import gradio as gr
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan,M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import soundfile as sf
import librosa
import tempfile
import os
import numpy as np
from huggingface_hub import login, hf_hub_download 
import re 


hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print("Logging in to Hugging Face Hub...")
    login(token=hf_token)
else:
    print("HF_TOKEN environment variable not set. Depending on model permissions, this might cause issues.")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading ASR (Whisper) model...")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small.en",
    device=device
)

print("Loading multilingual translation model (facebook/m2m100_418M)...")
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
translation_model = M2M100ForConditionalGeneration.from_pretrained(model_name).half().to(device)
translation_model.eval()
torch.set_grad_enabled(False)

def translate_text(text, target_lang):
    tokenizer.src_lang = "en"
    encoded = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = translation_model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id(target_lang)
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]


print("Loading TTS (SpeechT5) models...")
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

print("Loading speaker embeddings...")
try:
    embedding_path = hf_hub_download(
        repo_id="Matthijs/speecht5-tts-demo",
        filename="speaker_embedding.npy",
        repo_type="space"
    )
    speaker_embeddings = torch.tensor(np.load(embedding_path)).unsqueeze(0).to(device)
    print("✅ Speaker embeddings loaded successfully.")
except Exception as e:
    print(f"Error downloading speaker embedding: {e}")
    print("Falling back to a random embedding.")
    speaker_embeddings = torch.randn((1, 512)).to(device)

print("All models loaded. Launching Gradio...")

def voice_to_voice(audio_file):
    """
    Main function to transcribe, translate, and synthesize speech.
    Updated for higher Whisper transcription accuracy.
    """
    try:
        if audio_file is None:
            gr.Warning("No audio recorded. Please record your voice.")
            return None, None, None, None

        if isinstance(audio_file, str):
            audio_array, sampling_rate = librosa.load(audio_file, sr=None, mono=True)
        else:
            sampling_rate, audio_array = audio_file
            audio_array = np.array(audio_array).astype(np.float32)

        audio_array = librosa.util.normalize(audio_array)
        audio_array, _ = librosa.effects.trim(audio_array, top_db=25)
        if sampling_rate != 16000:
            audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000

        chunk_length_s = 20 
        stride_s = 2     
        chunk_len = int(chunk_length_s * sampling_rate)
        stride = int(stride_s * sampling_rate)
        step = chunk_len - stride

        all_texts = []
        for start in range(0, len(audio_array), step):
            end = min(start + chunk_len, len(audio_array))
            chunk = audio_array[start:end]
            if len(chunk) < 1000:
                continue
            result = asr_pipeline(chunk)
            text_part = result.get("text", "").strip()
            if text_part:
                all_texts.append(text_part)

        text = " ".join(all_texts).strip()
        if not text:
            gr.Warning("No clear speech detected. Please speak again.")
            return None, None, None, None

        print(f"✅ Transcribed Text: {text}")

        es_text, tr_text, zh_text = text_translation(text)
        print(f"Spanish Translation: {es_text}")
        print(f"Turkish Translation: {tr_text}")
        print(f"Chinese Translation: {zh_text}")

        es_audio_path = text_to_speech(es_text)
        tr_audio_path = text_to_speech(tr_text)
        zh_audio_path = text_to_speech(zh_text)

        return text, es_audio_path, tr_audio_path, zh_audio_path

    except Exception as e:
        print(f"Error in voice_to_voice: {e}")
        raise gr.Error(f"An error occurred: {e}")


def text_translation(text):
    """
    Translates the given text into Spanish, Turkish, and Chinese.
    """
    es_text = translate_text(text, "es")
    tr_text=translate_text(text, "tr")
    zh_text=translate_text(text, "zh")
    return es_text, tr_text, zh_text


def text_to_speech(text_to_speak):
    """
    Converts text to speech, handling long inputs by chunking.
    """
    try:
        chunks = re.split(r'(?<=[.!?]) +', text_to_speak)
        max_chunk_len = 450 
        processed_chunks = []

        for chunk in chunks:
            if len(chunk) > max_chunk_len:
                sub_chunks = [chunk[i:i+max_chunk_len] for i in range(0, len(chunk), max_chunk_len)]
                processed_chunks.extend(sub_chunks)
            else:
                processed_chunks.append(chunk)

        audio_pieces = []
        for i, chunk in enumerate(processed_chunks):
            if not chunk.strip():
                continue
            
            print(f"Generating audio for chunk {i+1}/{len(processed_chunks)}: '{chunk}'")
            inputs = tts_processor(text=chunk, return_tensors="pt")

            speech = tts_model.generate_speech(
                inputs["input_ids"].to(device),
                speaker_embeddings,
                vocoder=vocoder
            )
            audio_pieces.append(speech.cpu().numpy())

        if not audio_pieces:
            raise ValueError("No audio was generated. The text might be empty.")

        
        full_audio = np.concatenate(audio_pieces)

        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, full_audio, samplerate=16000)
            return f.name 

    except Exception as e:
        print(f"Error in text_to_speech for text '{text_to_speak}': {e}")
        raise gr.Error(f"Error generating audio with SpeechT5: {e}")


audio_input = gr.Microphone(
    label="🎙️ Speak (English)",
    type="numpy",   
    streaming=False,        
    show_label=True        
)


demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[
        gr.Textbox(label="Original Transcribed Text"),
        gr.Audio(label="Spanish Translation"),
        gr.Audio(label="Turkish Translation"),
        gr.Audio(label="Chinese Translation")
    ],
    title="Voice-to-Voice Translator (Hugging Face Edition)",
    description="Speak in English and get voice translations using 100% Hugging Face models (Whisper, Opus-MT, and SpeechT5).",
    flagging_mode=None 
)


if __name__ == "__main__":
    demo.launch(share=True)
