import gradio as gr
import whisper
from translate import Translator
from dotenv import dotenv_values
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

config = dotenv_values(".env")
ELEVENLABS_API_KEY = config["ELEVENLABS_API_KEY"]

translation_langs = ["en", "it", "fr", "ja"]

def translator(audio_file):
    # https://github.com/openai/whisper
    # Alternativa API Online: https://www.assemblyai.com/
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_file, language="es", fp16=False)
        original_transcription = result["text"]
    except Exception as  e:
        raise gr.Error(f"Se ha producido un error transcribiendo el texto: {str(e)}")

    # https://github.com/terryyin/translate-python

    try:
        transcriptions = []
        for lang in translation_langs:
            transcriptions.append(
                Translator(
                    from_lang="es", 
                    to_lang=lang
                ).translate(original_transcription)
            )
    except Exception as e:
        raise gr.Error(f"Se ha producido un error traduciendo el texto: {str(e)}")

    file_paths = []
    for idx, transcription in enumerate(transcriptions):
        lang = translation_langs[idx]
        file_paths.append(text_to_speach(transcription, lang))

    return file_paths



def text_to_speach(text: str, language: str) -> str:

    try:
        client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        response = client.text_to_speech.convert(
            voice_id="pNInz6obpgDQGcFmaJgB",
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_turbo_v2",
            voice_settings=VoiceSettings(
                stability=0.0,
                similarity_boost=0.0,
                style=0.0,
                use_speaker_boost=True,
            ),
        )
        save_file_path = f"audios/{language}.mp3"

        with open(save_file_path, "wb") as f:
            for chunk in response:
                if chunk:
                    f.write(chunk)

    except Exception as e:
        raise gr.Error(
            f"Se ha producido un error creando el audio: {str(e)}")

    return save_file_path


web  = gr.Interface(
    fn=translator,
    inputs=gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="Español"
    ),
    outputs=[
        gr.Audio(label="Inglés"),
        gr.Audio(label="Italiano"),
        gr.Audio(label="Francés"),
        gr.Audio(label="Japonés")
    ],
    title="Traductor de voz",
    description="Traductor de voz con IA a varios idiomas"
)

web.launch()