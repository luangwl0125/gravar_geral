# 🎙️ Assistente de Organização

Sistema de gravação, transcrição e organização de áudio para reuniões, palestras, atendimentos e outros eventos.

## 📋 Descrição

Este projeto oferece uma interface web amigável para:
- Gravação de áudio via microfone
- Upload de arquivos de áudio/vídeo
- Transcrição automática usando OpenAI Whisper
- Análise e organização do conteúdo usando ChatGPT
- Armazenamento local das transcrições e análises

## 🚀 Funcionalidades

- **Gravação ao Vivo**: Capture áudio diretamente do microfone
- **Processamento de Arquivos**: Suporte para diversos formatos de áudio e vídeo
- **Transcrição Inteligente**: Conversão precisa de fala para texto
- **Análise Automática**: Estruturação e organização do conteúdo
- **Interface Web**: Fácil de usar através do Streamlit
- **Armazenamento Local**: Todas as transcrições são salvas localmente

## 🛠️ Requisitos

- Python 3.11 ou superior
- Chave de API da OpenAI
- Microfone (para gravação ao vivo)
- Navegador web moderno

## ⚙️ Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-usuario/gravar_geral.git
cd gravar_geral
```

2. Crie e ative o ambiente virtual:
```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente:
   - Crie um arquivo `.streamlit/secrets.toml`
   - Adicione sua chave da API OpenAI:
     ```toml
     OPENAI_API_KEY = "sua-chave-aqui"
     ```

## 🎯 Uso

1. Ative o ambiente virtual:
```bash
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. Execute o aplicativo:
```bash
streamlit run gravar.py
```

3. Acesse a interface web em `http://localhost:8501`

## 📁 Estrutura do Projeto

```
gravacao_geral/
├── .gitignore
├── requirements.txt
├── .env                # variáveis de ambiente
├── gravar.py          # script principal
├── README.md          # este arquivo
├── temp/              # arquivos temporários  
│   └── *.wav
└── TRANSCRICOES/      # transcrições e análises
    ├── *_transcricao.txt
    └── *_analise.txt
```

## 🔒 Segurança

- Nunca compartilhe sua chave da API OpenAI
- Mantenha o arquivo `secrets.toml` no `.gitignore`
- Revise as transcrições antes de compartilhar

## 🤝 Contribuição

Contribuições são bem-vindas! Por favor, siga estas etapas:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## ✨ Agradecimentos

- OpenAI pelo Whisper e ChatGPT
- Streamlit pela excelente framework
- Todos os contribuidores de bibliotecas open source utilizadas