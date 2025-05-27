# RAG 통합 ChatLLM

## 설명

이 프로젝트는 사용자가 대규모 언어 모델(LLM)과 상호 작용할 수 있는 웹 기반 채팅 애플리케이션입니다. 스트리밍 응답, 채팅 기록, LLM 메시지의 Markdown 렌더링 기능을 제공하며, Qdrant를 벡터 데이터베이스로 사용하는 검색 증강 생성(RAG) 시스템을 갖추고 있습니다. 사용자는 Qdrant 데이터베이스에 문서를 업로드할 수 있으며, 애플리케이션은 이 문서에서 관련 정보를 검색하여 LLM에게 보다 문맥에 맞는 정확한 답변을 제공하도록 합니다.

## 주요 기능

*   **대화형 채팅 인터페이스:** 깔끔하고 현대적인 UI의 채팅 환경.
*   **스트리밍 LLM 응답:** LLM의 메시지가 토큰 단위로 실시간 표시됩니다.
*   **Markdown 렌더링:** LLM 응답이 Markdown으로 렌더링되어 서식 있는 텍스트, 코드 블록 등을 표현할 수 있습니다.
*   **채팅 기록:** 대화 기록이 유지되며 LLM에 컨텍스트로 전송됩니다.
*   **검색 증강 생성 (RAG):**
    *   벡터 데이터베이스로 **Qdrant**를 사용합니다.
    *   텍스트 임베딩 생성을 위해 **Sentence Transformers** (`all-MiniLM-L6-v2`)를 사용합니다.
    *   인덱싱을 위한 텍스트 문서 업로드 엔드포인트 (`/upload_document`).
    *   LLM 프롬프트를 보강하여 더 정보에 입각한 답변을 제공하기 위해 관련 문서 스니펫을 검색합니다.
*   **생성 중지:** "중지" 버튼으로 사용자가 LLM의 응답 스트림을 중단할 수 있습니다.
*   **FastAPI 백엔드:** 강력하고 비동기적인 Python 백엔드.
*   **Vanilla JavaScript 프론트엔드:** 무거운 프레임워크 없이 간단하고 효율적인 프론트엔드.

## 설정 및 설치

애플리케이션을 설정하고 실행하려면 다음 단계를 따르십시오:

**1. Python 가상 환경:**

의존성 관리를 위해 Python 가상 환경을 사용하는 것을 강력히 권장합니다.

```bash
# 가상 환경 생성 (예: .venv)
python3 -m venv .venv

# 가상 환경 활성화
# macOS 및 Linux:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate
```

**2. 의존성 설치:**

`pip`를 사용하여 필요한 Python 패키지를 설치합니다:

```bash
pip install -r requirements.txt
```
주요 의존성에는 `fastapi`, `uvicorn`, `qdrant-client`, `sentence-transformers`가 포함됩니다. 뒤의 두 가지는 RAG 기능에 매우 중요합니다.

**⚠️ RAG 의존성에 대한 중요 참고 사항:**
제공된 환경에서 개발 중, 의존성 설치 시, 특히 `torch` (`sentence-transformers`의 하위 의존성) 설치 과정에서 "No space left on device" (장치에 남은 공간 없음) 오류가 발생했습니다. 이로 인해 모든 RAG 관련 패키지의 완전하고 깨끗한 설치가 불가능했습니다.
**유사한 문제가 발생하면 RAG 기능(문서 업로드 및 컨텍스트 검색)이 올바르게 작동하지 않거나 전혀 작동하지 않을 수 있습니다.** 사용 환경에 충분한 디스크 공간이 있는지, 그리고 `requirements.txt`의 모든 패키지를 성공적으로 설치할 수 있는지 확인하십시오.

**3. Qdrant 벡터 데이터베이스 설정:**

RAG 시스템에는 실행 중인 Qdrant 인스턴스가 필요합니다. 애플리케이션은 `localhost:6333`에서 Qdrant에 연결하도록 구성되어 있습니다.

Qdrant를 실행하는 가장 쉬운 방법은 Docker를 사용하는 것입니다:

```bash
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage \
    qdrant/qdrant
```
이 명령은 영구 저장을 위해 로컬 디렉토리(`qdrant_storage`)를 마운트합니다.

**4. LLM 서버:**

이 애플리케이션은 채팅 완성을 위한 OpenAI API 형식과 호환되는 별도의 LLM 서버가 필요합니다.
*   애플리케이션은 `main.py`에서 `LLM_API_URL = "http://localhost:1234/v1/chat/completions"`의 LLM API에 연결하도록 구성되어 있습니다.
*   기본 모델은 `LLM_MODEL_NAME = "gemma-3-27b-it"`입니다.

**LM Studio** 또는 **Ollama (OpenAI 호환 프록시/어댑터 사용)**와 같은 도구를 사용하여 로컬 LLM을 제공할 수 있습니다. `LLM_MODEL_NAME`에 지정된 모델이 로드되어 있고 서버의 OpenAI 호환 엔드포인트를 통해 액세스할 수 있는지 확인하십시오.

## 애플리케이션 실행

설정이 완료되면 다음을 수행합니다:

1.  Python 가상 환경이 활성화되어 있는지 확인합니다.
2.  Qdrant 인스턴스가 실행 중인지 확인합니다.
3.  LLM 서버가 실행 중이고 구성되어 있는지 확인합니다.
4.  Uvicorn을 사용하여 FastAPI 애플리케이션을 실행합니다:

    ```bash
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
    ```
    *   `--reload`: 코드 변경 시 자동 재로드를 활성화합니다 (개발용).
    *   `--host 0.0.0.0`: 로컬 네트워크에서 서버에 액세스할 수 있도록 합니다.
    *   `--port 8000`: 포트를 지정합니다. 필요한 경우 변경할 수 있습니다.

## 사용 방법

**1. 웹 인터페이스 접속:**

웹 브라우저를 열고 `http://localhost:8000` (또는 구성한 포트)으로 이동합니다.

**2. 채팅:**

*   채팅 인터페이스 하단의 입력 상자에 메시지를 입력합니다.
*   Enter 키를 누르거나 "전송" 버튼을 클릭합니다.
*   LLM의 응답이 채팅 디스플레이로 스트리밍됩니다.

**3. 생성 중지:**

*   LLM이 긴 응답을 생성하는 경우 "중지" 버튼을 클릭하여 중단할 수 있습니다.

**4. 문서 업로드 (RAG용):**

RAG 의존성(`qdrant-client`, `sentence-transformers` 및 `torch`와 같은 하위 의존성)이 성공적으로 설치되었고 Qdrant가 실행 중인 경우, 컨텍스트 검색에 사용할 문서를 업로드할 수 있습니다.

`curl`이나 Postman과 같은 도구를 사용하여 `/upload_document` 엔드포인트로 POST 요청을 보냅니다.

**`curl` 사용 예시:**

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"text_content": "인공 지능(AI)은 다양한 산업을 빠르게 변화시키고 있습니다. AI의 하위 집합인 머신 러닝은 데이터를 기반으로 알고리즘을 학습시켜 예측이나 결정을 내립니다."}' \
     http://localhost:8000/upload_document
```

응답:
```json
{
  "message": "Document uploaded and indexed successfully.",
  "doc_id": "your-unique-document-id"
}
```

문서가 업로드되면 채팅 시스템은 사용자의 메시지를 기반으로 이러한 문서에서 관련 스니펫을 자동으로 찾아 LLM에 컨텍스트로 제공합니다.

## 파일 구조

*   `main.py`: FastAPI 백엔드 애플리케이션. 모든 API 엔드포인트, RAG 로직, LLM 상호작용 및 Qdrant 설정을 포함합니다.
*   `templates/index.html`: 채팅 인터페이스를 위한 기본 HTML 파일. 스타일링을 위한 CSS와 프론트엔드 로직을 위한 JavaScript를 포함합니다.
*   `static/marked.min.js`: 채팅창에 Markdown을 렌더링하는 데 사용되는 `marked.js` 라이브러리. (참고: 이 파일은 별도로 제공되거나 구해야 합니다).
*   `requirements.txt`: 프로젝트의 Python 의존성 목록입니다.
*   `README.md`: 이 파일, 프로젝트 문서를 제공합니다.
*   `qdrant_storage/` (선택 사항, Docker에 의해 생성됨): 위의 Docker 명령을 사용하는 경우 이 디렉토리에 Qdrant 데이터가 저장됩니다.

---
이 `README.md`는 애플리케이션 설정, 실행 및 사용에 대한 포괄적인 가이드를 제공합니다.
