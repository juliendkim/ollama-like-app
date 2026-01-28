# Gemma 3 Standalone Chatbot

<details>
<summary><strong>🇰🇷 Korean Version / 한국어 버전</strong></summary>

# Gemma 3 스탠드얼론 챗봇

구글의 **Gemma 3 1B** 모델을 기반으로 한 경량 스탠드얼론 챗봇 애플리케이션입니다. 이 프로젝트는 **FastAPI**를 사용하여 거대 언어 모델(LLM)을 서빙하고, 커맨드라인 인터페이스(CLI)를 통해 모델과 상호작용하는 방법을 보여줍니다.

**NVIDIA GPU (CUDA)** 및 **Apple Silicon (MPS)** 하드웨어 가속을 지원하며, 가속기가 없는 경우 CPU 모드로 작동합니다.

## ✨ 주요 기능

- **로컬 인퍼런스**: 외부 API 호출 없이 로컬 장비에서 직접 모델을 구동합니다.
- **RESTful API**: FastAPI를 사용하여 구축된 확장 가능한 백엔드 서버입니다.
- **CLI 클라이언트**: 사용하기 쉬운 터미널 기반 채팅 인터페이스를 제공합니다.
- **하드웨어 최적화**:
    - NVIDIA GPU (CUDA) 지원
    - Apple Silicon (MPS - Metal Performance Shaders) 지원
    - 자동 디바이스 감지 및 `float16` 정밀도 최적화
- **대화 기억**: 멀티턴 대화를 위한 컨텍스트 관리 기능을 포함합니다.

## 🛠️ 사전 요구 사항

- **Python 3.10** 이상
- **Hugging Face 계정** 및 **Access Token** (모델 다운로드용)
- (권장) NVIDIA GPU 또는 Apple Silicon Mac

## 🚀 설치 및 설정

### 1. 환경 설정

먼저 필요한 패키지를 설치합니다. 가상 환경 사용을 권장합니다.

```bash
# 가상 환경 생성 및 활성화 (예시)
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate  # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 2. Hugging Face 토큰 설정

`.env` 파일을 생성하고 Hugging Face 토큰을 입력합니다. (참고: `.env.example` 파일이 있다면 복사해서 사용하세요)

```bash
# .env 파일 생성
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

### 3. 모델 다운로드

스크립트를 실행하여 Hugging Face Hub에서 Gemma 3 모델을 다운로드합니다.

```bash
python download-model.py
```
> **참고**: 모델 크기는 약 2~4GB이며, 인터넷 속도에 따라 시간이 소요될 수 있습니다.

## 💻 사용 방법

### 0. 서버 실행 전 확인사항
- 다운로드 된 그대로 이용
    - modles/models--google--gemma-3-1b-it/snapshots/<HASH> 폴더가 있는지 확인
    - api.py 파일의 model_name을 "models/models--google--gemma-3-1b-it/snapshots/<HASH>"로 변경
- symbolic link를 실제 파일로 변경 후 이용
    - modles/models--google--gemma-3-1b-it/snapshots/<HASH> 폴더의 symbolic link를 실제 파일로 변경 후, models/gemma3-1b-it로 변경

### 1. API 서버 실행

백엔드 서버를 시작합니다.

```bash
./run.sh
# 또는
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
서버가 시작되면 `http://localhost:8000`에서 대기합니다.

### 2. 채팅 클라이언트 실행

새로운 터미널 창을 열고 클라이언트를 실행합니다.

```bash
python chat.py
```

### 3. API 문서

서버가 실행 중일 때 브라우저에서 아래 주소로 접속하면 API 문서를 확인할 수 있습니다.
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 📁 프로젝트 구조

- `api.py`: FastAPI 백엔드 서버 및 모델 로딩 로직
- `chat.py`: 사용자와 상호작용하는 CLI 클라이언트
- `download-model.py`: 모델 다운로드 유틸리티
- `run.sh`: 서버 실행 스크립트
- `models/`: 다운로드된 모델이 저장되는 디렉토리

## 📄 라이선스

이 프로젝트는 MIT 라이선스에 따라 라이선스가 부과됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하십시오.

</details>

---

A lightweight, standalone chatbot application powered by Google's **Gemma 3 1B** model. This project demonstrates how to serve a Large Language Model (LLM) using **FastAPI** and interact with it via a command-line interface (CLI).

It supports hardware acceleration on **NVIDIA GPUs (CUDA)** and **Apple Silicon (MPS)**, falling back to CPU if neither is available.

---

## ✨ Features

- **Local Inference**: Runs entirely on your machine; no external API calls required after download.
- **RESTful API**: Scalable backend built with FastAPI.
- **CLI Client**: Easy-to-use terminal-based chat interface.
- **Hardware Optimization**:
    - Supports NVIDIA GPUs (CUDA).
    - Supports Apple Silicon (MPS - Metal Performance Shaders).
    - Automatic device detection and `float16` precision optimization.
- **Conversation History**: Manages context for multi-turn conversations.

## 🛠️ Prerequisites

- **Python 3.10** or higher.
- **Hugging Face Account** and **Access Token** (to download the model).
- (Recommended) NVIDIA GPU or Apple Silicon Mac.

## 🚀 Installation & Setup

## 💻 Usage

### 0. Before Running the Server
- **Use as downloaded**:
    - Verify that the `models/models--google--gemma-3-1b-it/snapshots/<HASH>` folder exists.
    - Update `model_name` in `api.py` to `"models/models--google--gemma-3-1b-it/snapshots/<HASH>"`.
- **Use after converting symbolic links to actual files**:
    - Replace symbolic links in `models/models--google--gemma-3-1b-it/snapshots/<HASH>` with actual files.
    - Rename the directory to `models/gemma3-1b-it`.


### 1. Environment Setup

Install the required packages. Using a virtual environment is recommended.

```bash
# Create and activate virtual environment (example)
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Hugging Face Token

Create a `.env` file and add your Hugging Face token.

```bash
# Create .env file
echo "HUGGINGFACE_TOKEN=your_token_here" > .env
```

### 3. Download Model

Run the script to download the Gemma 3 model from Hugging Face Hub.

```bash
python download-model.py
```
> **Note**: The model is approximately 2-4GB. Download time depends on your internet connection.

## 💻 Usage

### 1. Start the API Server

Launch the backend server.

```bash
./run.sh
# OR
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
The server will start listening at `http://localhost:8000`.

### 2. Start the Chat Client

Open a new terminal window and run the client.

```bash
python chat.py
```

### 3. API Documentation

Once the server is running, you can access the interactive API documentation at:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 📁 Project Structure

- `api.py`: FastAPI backend server and model loading logic.
- `chat.py`: CLI client for user interaction.
- `download-model.py`: Utility to download the model.
- `run.sh`: Server startup script.
- `models/`: Directory where the downloaded model is stored.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
