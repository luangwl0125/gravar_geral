import os
import warnings
from pathlib import Path
from datetime import datetime
import time
import queue
import tempfile

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import openai
from openai import RateLimitError, OpenAIError
import pydub
from pydub.exceptions import CouldntDecodeError
from moviepy.video.io.VideoFileClip import VideoFileClip
from dotenv import load_dotenv, find_dotenv
import whisper
import torch

# ——— Configurações iniciais ———
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning, module="audioop")
if hasattr(torch, "classes"):
    torch.classes.__path__ = []

# Carrega variáveis de ambiente
_ = load_dotenv(find_dotenv())

# Pastas de trabalho
BASE_DIR = Path(__file__).parent
PASTA_TEMP = BASE_DIR / "temp"
PASTA_TRANSCRICOES = BASE_DIR / "TRANSCRICOES"
PASTA_TEMP.mkdir(exist_ok=True)
PASTA_TRANSCRICOES.mkdir(exist_ok=True)

# Cliente OpenAI e modelo Whisper local
client = openai.OpenAI()
_local_whisper = None
def get_local_whisper():
    global _local_whisper
    if _local_whisper is None:
        _local_whisper = whisper.load_model("base")
    return _local_whisper

# Retry config
MAX_RETRIES = 3
RETRY_DELAY = 2  # segundos

def handle_openai_error(func):
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    st.warning("Rate limit atingido; usando Whisper local.")
                    return use_fallback(*args, **kwargs)
                time.sleep(RETRY_DELAY * (attempt + 1))
            except OpenAIError as e:
                st.error(f"Erro na API OpenAI: {e}")
                return None, None
            except Exception as e:
                st.error(f"Erro inesperado: {e}")
                return None, None
    return wrapper

def use_fallback(caminho_audio=None, prompt=None, texto=None):
    """Fallback usando Whisper local"""
    try:
        if caminho_audio:
            res = get_local_whisper().transcribe(caminho_audio, language="pt")
            return res["text"], "Análise indisponível (fallback local)"
        if texto:
            return texto, "Análise indisponível (fallback local)"
    except Exception as e:
        st.error(f"Fallback falhou: {e}")
    return "", ""

PROMPT_ANALISE = (
"""
Você é um Assistente de Organização Profissional. Estruture a transcrição em:

- IDENTIFICAÇÃO DO EVENTO  
- DEMANDA PRINCIPAL  
- PONTOS DISCUTIDOS  
- DECISÕES TOMADAS  
- OBSERVAÇÕES  
- SUGESTÕES  

#### TRANSCRIÇÃO ####
{}
#### TRANSCRIÇÃO ####
"""
)

@handle_openai_error
def processa_transcricao_chatgpt(texto: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": PROMPT_ANALISE.format(texto)}]
    )
    return resp.choices[0].message.content

def converter_para_wav(caminho: str) -> str:
    try:
        audio = pydub.AudioSegment.from_file(caminho)
    except CouldntDecodeError:
        st.error("Formato de áudio não suportado.")
        return ""
    fd, wav_path = tempfile.mkstemp(suffix=".wav", prefix="conv_")
    os.close(fd)
    audio.export(wav_path, format="wav")
    return wav_path

@handle_openai_error
def transcreve_audio(caminho_audio: str, prompt: str) -> tuple[str, str]:
    texto = analise = ""
    try:
        with open(caminho_audio, "rb") as f:
            resp = client.audio.transcriptions.create(
                model="whisper-1",
                language="pt",
                response_format="text",
                file=f,
                prompt=prompt or None
            )
            texto = resp
            analise = processa_transcricao_chatgpt(texto)
    except Exception:
        st.warning("Erro na API; usando fallback local.")
        texto, analise = use_fallback(caminho_audio, prompt)
    finally:
        # Cleanup de temporário
        if caminho_audio.startswith(tempfile.gettempdir()):
            try: os.remove(caminho_audio)
            except OSError: pass
    return texto, analise

# — Parâmetros de sessão inicial —
st.session_state.setdefault("transcricao_mic", "")
st.session_state.setdefault("analise_mic", "")
st.session_state.setdefault("gravando_audio", False)
st.session_state.setdefault("audio_completo", pydub.AudioSegment.empty())

@st.cache_data
def get_ice_servers():
    return [{"urls": ["stun:stun.l.google.com:19302"]}]

def adiciona_chunk(frames):
    seg = pydub.AudioSegment.empty()
    for frame in frames:
        seg += pydub.AudioSegment(
            data=frame.to_ndarray().tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels),
        )
    return seg

def salva_transcricao(texto, analise, origem=""):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{ts}_{origem}" if origem else ts
    (PASTA_TRANSCRICOES / f"{base}_txt.txt").write_text(texto, encoding="utf-8")
    (PASTA_TRANSCRICOES / f"{base}_ana.txt").write_text(analise, encoding="utf-8")

# — Abas do Streamlit —
def transcreve_tab_mic():
    prompt = st.text_input("Prompt (opcional)", key="pmic")
    btn = "⏹️ Parar" if st.session_state["gravando_audio"] else "🔴 Gravar"
    if st.button(btn):
        st.session_state["gravando_audio"] = not st.session_state["gravando_audio"]
        if not st.session_state["gravando_audio"]:
            fn = PASTA_TRANSCRICOES / f"mic_{datetime.now():%Y%m%d_%H%M%S}.wav"
            st.session_state["audio_completo"].export(fn, format="wav")
            st.session_state["audio_completo"] = pydub.AudioSegment.empty()

    ctx = webrtc_streamer(
        key="mic", mode=WebRtcMode.SENDONLY,
        media_stream_constraints={"video": False, "audio": True},
        rtc_configuration={"iceServers": get_ice_servers()},
    )
    if not ctx.state.playing and st.session_state["transcricao_mic"]:
        if not st.session_state["analise_mic"]:
            st.write("Gerando análise...")
            st.session_state["analise_mic"] = processa_transcricao_chatgpt(
                st.session_state["transcricao_mic"]
            )
        st.markdown("**Transcrição:**"); st.write(st.session_state["transcricao_mic"])
        st.markdown("**Análise:**");     st.write(st.session_state["analise_mic"])
        salva_transcricao(
            st.session_state["transcricao_mic"],
            st.session_state["analise_mic"],
            "mic",
        )
        return

    placeholder = st.empty()
    buffer = pydub.AudioSegment.empty()
    ultimo = time.time()
    st.session_state["transcricao_mic"] = ""
    st.session_state["analise_mic"] = ""

    while ctx.audio_receiver:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            time.sleep(0.1); continue

        seg = adiciona_chunk(frames)
        buffer += seg
        if st.session_state["gravando_audio"]:
            st.session_state["audio_completo"] += seg

        if time.time() - ultimo > 10 and buffer:
            ultimo = time.time()
            tmp = PASTA_TEMP / "mic_tmp.wav"
            buffer.export(str(tmp), format="wav")
            txt, _ = transcreve_audio(str(tmp), prompt)
            st.session_state["transcricao_mic"] += txt
            placeholder.write(st.session_state["transcricao_mic"])
            buffer = pydub.AudioSegment.empty()

def _extrai_audio_de_video(file):
    tmp_vid = PASTA_TEMP / "vid_tmp.mp4"
    tmp_vid.write_bytes(file.read())
    clip = VideoFileClip(str(tmp_vid))
    out = PASTA_TEMP / "vid_tmp.wav"
    clip.audio.write_audiofile(str(out), logger=None)
    return str(out)

def transcreve_tab_video():
    prompt = st.text_input("Prompt (opcional)", key="pvid")
    vid = st.file_uploader("Vídeo", type=["mp4","mov","avi","mkv","webm"])
    if not vid: return
    wav_in = _extrai_audio_de_video(vid)
    wav_conv = converter_para_wav(wav_in)
    txt, ana = transcreve_audio(wav_conv, prompt)
    st.markdown("**Transcrição:**"); st.write(txt)
    st.markdown("**Análise:**");     st.write(ana)
    salva_transcricao(txt, ana, f"vid_{vid.name}")

def transcreve_tab_audio():
    prompt = st.text_input("Prompt (opcional)", key="paud")
    aud = st.file_uploader("Áudio", type=["mp3","wav","m4a","opus","mpeg"])
    if not aud: return
    path = PASTA_TEMP / aud.name
    path.write_bytes(aud.read())
    wav = converter_para_wav(str(path))
    txt, ana = transcreve_audio(wav, prompt)
    st.markdown("**Transcrição:**"); st.write(txt)
    st.markdown("**Análise:**");     st.write(ana)
    salva_transcricao(txt, ana, f"aud_{aud.name}")

def transcreve_tab_texto():
    st.write("Envie .txt ou .docx para análise")
    txtfile = st.file_uploader("Texto", type=["txt","docx"])
    if not txtfile: return
    if txtfile.type == "text/plain":
        txt = txtfile.getvalue().decode("utf-8")
    else:
        import docx2txt
        txt = docx2txt.process(txtfile)
    ana = processa_transcricao_chatgpt(txt)
    st.markdown("**Original:**"); st.write(txt)
    st.markdown("**Análise:**");  st.write(ana)
    salva_transcricao(txt, ana, f"txt_{txtfile.name}")

# — Função principal —
def main():
    st.header("🎙️ Assistente de Organização 🎙️")
    tabs = st.tabs(["Microfone","Vídeo","Áudio","Texto"])
    with tabs[0]: transcreve_tab_mic()
    with tabs[1]: transcreve_tab_video()
    with tabs[2]: transcreve_tab_audio()
    with tabs[3]: transcreve_tab_texto()

if __name__ == "__main__":
    main()
