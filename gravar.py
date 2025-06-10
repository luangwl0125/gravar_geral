import os
import warnings

# 1. Desativa o file watcher do Streamlit (evita erros com PyTorch)
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# 2. Suprime o DeprecationWarning do módulo audioop
warnings.filterwarnings("ignore", category=DeprecationWarning, module="audioop")

from pathlib import Path
from datetime import datetime
import time
import queue
import tempfile

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

import openai
import pydub
from moviepy.video.io.VideoFileClip import VideoFileClip
from dotenv import load_dotenv, find_dotenv
import whisper
import torch
from openai import RateLimitError

# 3. Corrige o erro "Tried to instantiate class '__path__._path'"
if hasattr(torch, "classes"):
    torch.classes.__path__ = []

# Carrega variáveis de ambiente
_ = load_dotenv(find_dotenv())

# Diretório temporário
PASTA_TEMP = Path(__file__).parent / 'temp'
PASTA_TEMP.mkdir(exist_ok=True)

# Diretório de transcrições
PASTA_TRANSCRICOES = Path(__file__).parent / 'TRANSCRICOES'
PASTA_TRANSCRICOES.mkdir(exist_ok=True)

# Arquivos temporários
ARQUIVO_AUDIO_TEMP = PASTA_TEMP / 'audio.wav'
ARQUIVO_VIDEO_TEMP = PASTA_TEMP / 'video.mp4'
ARQUIVO_MIC_TEMP = PASTA_TEMP / 'mic.wav'

# Cliente OpenAI
client = openai.OpenAI()

# Modelo Whisper local para fallback
local_model = None
def get_local_whisper():
    global local_model
    if local_model is None:
        local_model = whisper.load_model("base")
    return local_model

# Configurações de retry
MAX_RETRIES = 3
RETRY_DELAY = 2  # segundos

def handle_openai_error(func):
    def wrapper(*args, **kwargs):
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except RateLimitError:
                if attempt == MAX_RETRIES - 1:
                    st.warning("OpenAI API rate limit atingido. Usando serviço local de fallback.")
                    return use_fallback_service(*args, **kwargs)
                else:
                    time.sleep(RETRY_DELAY * (attempt + 1))
            except Exception as e:
                st.error(f"Erro ao processar: {e}")
                return None
    return wrapper

def use_fallback_service(caminho_audio=None, prompt=None, texto=None):
    """Serviço de fallback usando Whisper local"""
    try:
        if caminho_audio:
            model = get_local_whisper()
            result = model.transcribe(caminho_audio, language="pt")
            return result["text"], "Análise não disponível (serviço local)"
        elif texto:
            return texto, "Análise não disponível (serviço local)"
    except Exception as e:
        st.error(f"Erro no serviço de fallback: {e}")
        return "", ""

# Prompt para o ChatGPT
PROMPT_ANALISE = '''
Você é um Assistente de Organização Profissional. Sua tarefa é estruturar tecnicamente a transcrição de um evento gravado — que pode ser uma reunião, aula, palestra, entrevista, atendimento ou similar — em uma análise organizada, clara e objetiva.

Siga estas diretrizes:
1. Utilize linguagem técnica e formal, adequada ao ambiente profissional.
2. Evite inferências indevidas. Se informações estiverem ausentes, indique como "não informado".
3. Estruture a resposta com as seções abaixo. Se não aplicável, mantenha o título e indique "não informado".

SEÇÕES OBRIGATÓRIAS:
- IDENTIFICAÇÃO DO EVENTO  
- OBJETIVO OU DEMANDA PRINCIPAL  
- PRINCIPAIS PONTOS DISCUTIDOS  
- AÇÕES REALIZADAS OU DECISÕES TOMADAS  
- ANÁLISE GERAL OU OBSERVAÇÕES  
- CONCLUSÃO E SUGESTÕES  

#### TRANSCRIÇÃO ####
{}
#### TRANSCRIÇÃO ####
'''

@handle_openai_error
def processa_transcricao_chatgpt(texto: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": PROMPT_ANALISE.format(texto)}]
    )
    return resp.choices[0].message.content

def converter_para_wav(caminho_entrada: str) -> str:
    audio = pydub.AudioSegment.from_file(caminho_entrada)
    fd, caminho_wav = tempfile.mkstemp(suffix=".wav", prefix="audio_")
    os.close(fd)
    audio.export(caminho_wav, format='wav')
    return caminho_wav

@handle_openai_error
def transcreve_audio(caminho_audio: str, prompt: str) -> tuple[str, str]:
    with open(caminho_audio, 'rb') as f:
        try:
            resp = client.audio.transcriptions.create(
                model='whisper-1',
                language='pt',
                response_format='text',
                file=f,
                prompt=prompt,
            )
            analise = processa_transcricao_chatgpt(resp)
            return resp, analise
        except Exception as e:
            st.warning(f"Erro na API OpenAI: {e}. Usando serviço local.")
            return use_fallback_service(caminho_audio, prompt)

# Estado inicial
if 'transcricao_mic' not in st.session_state:
    st.session_state['transcricao_mic'] = ''
if 'analise_mic' not in st.session_state:
    st.session_state['analise_mic'] = ''
if 'gravando_audio' not in st.session_state:
    st.session_state['gravando_audio'] = False
if 'audio_completo' not in st.session_state:
    st.session_state['audio_completo'] = pydub.AudioSegment.empty()

@st.cache_data
def get_ice_servers():
    return [{'urls': ['stun:stun.l.google.com:19302']}]

def adiciona_chunck_de_audio(frames, chunk_audio):
    for frame in frames:
        seg = pydub.AudioSegment(
            data=frame.to_ndarray().tobytes(),
            sample_width=frame.format.bytes,
            frame_rate=frame.sample_rate,
            channels=len(frame.layout.channels)
        )
        chunk_audio += seg
    return chunk_audio

def salva_transcricao(texto: str, analise: str, origem: str = ""):
    agora = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    prefixo = f"{agora}_{origem}" if origem else agora

    arq_txt = PASTA_TRANSCRICOES / f"{prefixo}_transcricao.txt"
    arq_ana = PASTA_TRANSCRICOES / f"{prefixo}_analise.txt"
    arq_txt.write_text(texto, encoding='utf-8')
    arq_ana.write_text(analise, encoding='utf-8')
    return arq_txt, arq_ana

def transcreve_tab_mic():
    prompt_mic = st.text_input('Prompt (opcional)', key='input_mic')
    col1, col2 = st.columns([3,1])
    with col2:
        if st.button('🔴 Gravar Áudio' if not st.session_state['gravando_audio'] else '⏹️ Parar Gravação'):
            st.session_state['gravando_audio'] = not st.session_state['gravando_audio']
            if not st.session_state['gravando_audio'] and len(st.session_state['audio_completo']) > 0:
                st.session_state['audio_completo'].export(
                    PASTA_TRANSCRICOES / f"audio_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.wav",
                    format='wav'
                )
                st.session_state['audio_completo'] = pydub.AudioSegment.empty()

    ctx = webrtc_streamer(
        key='mic', mode=WebRtcMode.SENDONLY,
        media_stream_constraints={'video': False, 'audio': True},
        rtc_configuration={"iceServers": get_ice_servers()}
    )

    if not ctx.state.playing:
        if st.session_state['transcricao_mic'] and not st.session_state['analise_mic']:
            st.write("Gerando análise...")
            st.session_state['analise_mic'] = processa_transcricao_chatgpt(st.session_state['transcricao_mic'])
        if st.session_state['transcricao_mic']:
            st.markdown("**Transcrição:**")
            st.write(st.session_state['transcricao_mic'])
            st.markdown("**Análise:**")
            st.write(st.session_state['analise_mic'])
            salva_transcricao(
                st.session_state['transcricao_mic'],
                st.session_state['analise_mic'],
                'microfone'
            )
        return

    placeholder = st.empty()
    chunk_audio = pydub.AudioSegment.empty()
    ultimo = time.time()
    st.session_state['transcricao_mic'] = ''
    st.session_state['analise_mic'] = ''

    while ctx.audio_receiver:
        try:
            frames = ctx.audio_receiver.get_frames(timeout=1)
        except queue.Empty:
            time.sleep(0.1)
            continue

        seg = pydub.AudioSegment.empty()
        seg = adiciona_chunck_de_audio(frames, seg)
        chunk_audio += seg
        if st.session_state['gravando_audio']:
            st.session_state['audio_completo'] += seg

        if time.time() - ultimo > 10 and len(chunk_audio) > 0:
            ultimo = time.time()
            chunk_audio.export(ARQUIVO_MIC_TEMP, format='wav')
            texto, _ = transcreve_audio(str(ARQUIVO_MIC_TEMP), prompt_mic)
            st.session_state['transcricao_mic'] += texto
            placeholder.write(st.session_state['transcricao_mic'])
            chunk_audio = pydub.AudioSegment.empty()

def _salva_audio_do_video(file_bytes):
    with open(ARQUIVO_VIDEO_TEMP, 'wb') as f:
        f.write(file_bytes.read())
    clip = VideoFileClip(str(ARQUIVO_VIDEO_TEMP))
    clip.audio.write_audiofile(str(ARQUIVO_AUDIO_TEMP), logger=None)

def transcreve_tab_video():
    prompt = st.text_input('Prompt (opcional)', key='input_video')
    video = st.file_uploader('Adicione um vídeo', type=['mp4','mov','avi','mkv','webm'])
    if video:
        _salva_audio_do_video(video)
        wav = converter_para_wav(str(ARQUIVO_AUDIO_TEMP))
        texto, analise = transcreve_audio(wav, prompt)
        st.markdown("**Transcrição:**")
        st.write(texto)
        st.markdown("**Análise:**")
        st.write(analise)
        salva_transcricao(texto, analise, f'video_{video.name}')

def transcreve_tab_audio():
    prompt = st.text_input('Prompt (opcional)', key='input_audio')
    audio = st.file_uploader('Adicione um áudio', type=['opus','mp4','mpeg','wav','mp3','m4a'])
    if audio:
        caminho = PASTA_TEMP / audio.name
        caminho.write_bytes(audio.read())
        wav = converter_para_wav(str(caminho))
        texto, analise = transcreve_audio(wav, prompt)
        st.markdown("**Transcrição:**")
        st.write(texto)
        st.markdown("**Análise:**")
        st.write(analise)
        salva_transcricao(texto, analise, f'audio_{audio.name}')

def transcreve_tab_texto():
    st.write("Envie um arquivo de texto com a transcrição para análise")
    arquivo_texto = st.file_uploader('Adicione um arquivo de texto', type=['txt','doc','docx'])
    if arquivo_texto:
        try:
            if arquivo_texto.type == 'text/plain':
                texto = arquivo_texto.getvalue().decode('utf-8')
            else:
                import docx2txt
                texto = docx2txt.process(arquivo_texto)
            analise = processa_transcricao_chatgpt(texto)
            st.markdown("**Texto Original:**")
            st.write(texto)
            st.markdown("**Análise:**")
            st.write(analise)
            salva_transcricao(texto, analise, f'texto_{arquivo_texto.name}')
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")

def main():
    st.header('🎙️ Assistente de Organização 🎙️')
    st.markdown('Gravação, Transcrição e Organização.')
    st.markdown('Reuniões, Palestras, Atendimentos e Outros.')
    abas = st.tabs(['Microfone', 'Vídeo', 'Áudio', 'Texto'])
    with abas[0]:
        transcreve_tab_mic()
    with abas[1]:
        transcreve_tab_video()
    with abas[2]:
        transcreve_tab_audio()
    with abas[3]:
        transcreve_tab_texto()

if __name__ == '__main__':
    main()
