# Agent Workflow Memory (AWM) 시스템 상세 분석

이 문서는 AWM 논문의 핵심인 **워크플로우 축적(Induction)**과 **활용(Utilization)** 메커니즘을 상세히 설명합니다.

---

## 목차

1. [개요](#1-개요)
2. [워크플로우 축적 (Workflow Induction)](#2-워크플로우-축적-workflow-induction)
   - [오프라인 유도](#21-오프라인-유도-offline-induction)
   - [온라인 유도](#22-온라인-유도-online-induction)
   - [중복 제거 전략](#23-중복-제거-전략)
3. [워크플로우 활용 (Workflow Utilization)](#3-워크플로우-활용-workflow-utilization)
   - [워크플로우 검색](#31-워크플로우-검색-retrieval)
   - [프롬프트 구성](#32-프롬프트-구성)
   - [토큰 예산 관리](#33-토큰-예산-관리)
4. [전체 파이프라인](#4-전체-파이프라인)
5. [코드 참조](#5-코드-참조)

---

## 1. 개요

AWM은 웹 에이전트가 작업을 수행하면서 **재사용 가능한 워크플로우 패턴**을 학습하고 축적하는 메모리 시스템입니다.

### 핵심 아이디어

```
구체적인 작업 경험 → 추상화된 워크플로우 → 새로운 작업에 활용
```

### 두 가지 운영 모드

| 모드 | 데이터 소스 | 사용 시점 |
|------|------------|----------|
| **오프라인 (Offline)** | 정답이 주석된 학습 데이터 | 학습 데이터가 있을 때 |
| **온라인 (Online)** | 에이전트의 과거 성공 경험 | 학습 데이터 없이 실시간으로 |

---

## 2. 워크플로우 축적 (Workflow Induction)

### 2.1 오프라인 유도 (Offline Induction)

학습 데이터에서 워크플로우를 추출하는 방식입니다.

#### 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    오프라인 유도 파이프라인                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐                                       │
│  │   학습 데이터     │  JSON 형식의 주석된 예제들              │
│  │   (train/*.json) │                                       │
│  └────────┬────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │      데이터 계층화                        │               │
│  │  domain → subdomain → website           │               │
│  │  예: Travel → Airlines → delta          │               │
│  └────────┬────────────────────────────────┘               │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │      예제 포맷팅 (format_examples)       │               │
│  │  Query + Actions 텍스트로 변환           │               │
│  └────────┬────────────────────────────────┘               │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │      프롬프트 구성                        │               │
│  │  [Instruction] + [One-shot] + [예제들]   │               │
│  └────────┬────────────────────────────────┘               │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │      GPT-4o 호출                         │               │
│  │  공통 패턴을 추상화된 워크플로우로 변환     │               │
│  └────────┬────────────────────────────────┘               │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │      후처리 (filter_workflows)           │               │
│  │  중복 제거, 포맷 정리                     │               │
│  └────────┬────────────────────────────────┘               │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │      workflow/{website}.txt 저장         │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 입력 예제 형식

```
## Query 1: Find flights from Seattle to New York on June 5th
Actions:
[link]  From Departure Airport or City Your Origin -> CLICK
[textbox]  Origin City or Airport -> TYPE: Seattle
[link]  SEA Seattle, WA -> CLICK
[link]  To Destination Airport or City Your Destination -> CLICK
[textbox]  Destination City or Airport -> TYPE: New York
[link]  NYC New York City Area Airports, NY -> CLICK
[combobox]  Trip Type:, changes will reload the page -> CLICK
[option]  One Way -> CLICK
[button]  Depart and Return Calendar -> CLICK
[link]  5 June 2023, Monday -> CLICK
[button]  done -> CLICK
[label]  Shop with Miles -> CLICK
[button]  SUBMIT -> CLICK

## Query 2: Check all available one way flights from Manhattan to Philadelphia on May 23rd
Actions:
[link]  From Departure Airport or City Your Origin -> CLICK
[textbox]  Origin City or Airport -> TYPE: Manhattan
[link]  MHK Manhattan Regl, USA -> CLICK
...
```

#### 출력 워크플로우 형식

```
## enter_flight_locations
Given that you are on the Delta flight booking page, this workflow enters the departure and destination city/airport for your flight.
[link]  From Departure Airport or City Your Origin -> CLICK
[textbox]  Origin City or Airport -> TYPE: {your-origin-city}
[link]  {best-popup-option} -> CLICK
[link]  To Destination Airport or City Your Destination -> CLICK
[textbox]  Destination City or Airport -> TYPE: {your-destination-city}
[link]  {best-popup-option} -> CLICK

## select_oneway_trip
Given that you are on the Delta flight booking page, this workflow changes the flight to be one-way.
[combobox]  Trip Type:, changes will reload the page -> CLICK
[option]  One Way -> CLICK

## select_date_for_travel
Given that you are on the Delta flight booking page, this workflow selects the travel date.
[button]  Depart and Return Calendar Use enter to open -> CLICK
[link]  {travel-date} -> CLICK
[button]  done -> CLICK
```

#### 프롬프트 구성 요소

**1. Instruction (지시문)**
```
Given a list of web navigation tasks, your task is to extract the common
workflows to solve these tasks.

Each given task contains a natural language instruction, and a series of
actions to solve the task. You need to find the repetitive subset of actions
across multiple tasks, and extract each of them out as a workflow.

Each workflow should be a commonly-reused sub-routine of the tasks.
Do not generate similar or overlapping workflows.
Each workflow should have at least two steps.
```

**2. One-shot 예시**
- Delta 항공 웹사이트의 4개 구체적 예제
- 이를 바탕으로 생성된 4개의 추상화된 워크플로우
- LLM이 "구체적 → 추상적" 변환 방법을 학습

**3. 실제 입력**
- 현재 웹사이트의 학습 예제들

---

### 2.2 온라인 유도 (Online Induction)

학습 데이터 없이 에이전트의 **성공한 경험**에서 워크플로우를 추출합니다.

#### 파이프라인

```
┌─────────────────────────────────────────────────────────────┐
│                    온라인 유도 파이프라인                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐                                       │
│  │   테스트 작업    │  워크플로우 없이 에이전트 실행           │
│  │   (1차 시도)    │                                       │
│  └────────┬────────┘                                       │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │      성공한 궤적 수집                     │               │
│  │  results/{task_id}.json                 │               │
│  └────────┬────────────────────────────────┘               │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │      궤적 추출 (get_trajectory)          │               │
│  │  환경 상태 + 액션 쌍으로 파싱             │               │
│  └────────┬────────────────────────────────┘               │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │      GPT-4o 호출                         │               │
│  │  성공 경험에서 공통 패턴 추출             │               │
│  └────────┬────────────────────────────────┘               │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │      워크플로우 메모리 업데이트           │               │
│  │  workflow/{website}.txt에 추가           │               │
│  └────────┬────────────────────────────────┘               │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────────────────────────┐               │
│  │      다음 배치 작업 실행                  │               │
│  │  업데이트된 워크플로우로 재시도           │               │
│  └─────────────────────────────────────────┘               │
│                    │                                        │
│                    ▼                                        │
│              [반복: 점진적 개선]                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 궤적 추출 코드

```python
def get_trajectory(path: str):
    """에이전트 실행 로그에서 (환경상태, 액션) 쌍 추출"""
    trajectory = []
    result = json.load(open(path, 'r'))
    for item in result:
        if not is_io_dict(item): continue
        step = {
            "env": "# " + item["input"][-1]["content"],  # HTML 상태
            "action": item["output"],                     # 수행한 액션
        }
        trajectory.append(step)
    return trajectory
```

---

### 2.3 중복 제거 전략

#### Template ID 기반 중복 제거

같은 의도 템플릿을 공유하는 작업들을 그룹화하고, 각 그룹에서 1개만 선택합니다.

```python
# 템플릿 ID로 그룹화
template_dict = {}
for f in file_dirs:
    template_id = config["intent_template_id"]
    if template_id not in template_dict:
        template_dict[template_id] = []
    template_dict[template_id].append(workflow)

# 각 그룹에서 1개만 샘플링
selected_workflows = random_group_sample(template_dict, 1)
```

#### 추상 궤적 기반 중복 제거

액션 시퀀스를 추상화하여 동일한 패턴을 가진 워크플로우를 제거합니다.

```python
def get_abstract_trajectory(action_list):
    """
    click('123', 'text') → click('123')
    fill('456', 'hello') → fill('456')
    구체적 값을 제거하고 액션 패턴만 추출
    """
    abstract = []
    for acts in action_list:
        for a in acts:
            action = a[:a.index("(")]  # 액션 타입
            arg = a[s+1: e]            # 첫 번째 인자만
            abstract.append(f"{action}({arg})")
    return '_'.join(abstract)  # "click('123')_fill('456')_click('789')"
```

---

## 3. 워크플로우 활용 (Workflow Utilization)

### 3.1 워크플로우 검색 (Retrieval)

작업과 관련된 워크플로우를 선택적으로 가져올 수 있습니다.

#### 검색 모드

| 모드 | 방식 | 설명 |
|------|------|------|
| **random** | 무작위 | 워크플로우를 랜덤하게 선택 |
| **semantic** | 의미론적 | FAISS + OpenAI 임베딩으로 유사도 기반 검색 |

#### 의미론적 검색 구현

```python
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

def build_memory(workflows: list[dict], memory_path: str):
    """워크플로우 벡터 인덱스 생성"""
    # OpenAI 임베딩 모델 사용
    embedding = OpenAIEmbeddings(model="text-embedding-ada-002")

    # 워크플로우 이름 + 설명을 벡터화
    texts = [f"{w['name']}\n{w['docstring']}" for w in workflows]

    # FAISS 인덱스 생성
    memory = FAISS.from_texts(
        texts=texts,
        embedding=embedding,
        metadatas=[{"name": i} for i in range(len(workflows))]
    )
    return memory

def get_ids_and_scores(memory, query: str, top_k: int):
    """쿼리와 가장 유사한 워크플로우 검색"""
    docs_and_similarities = memory.similarity_search_with_score(query, top_k)
    retrieved_ids, scores = [], []
    for doc, score in docs_and_similarities:
        retrieved_ids.append(doc.metadata["name"])
        scores.append(score)
    return retrieved_ids, scores
```

#### 검색 예시

```
쿼리: "Find flights from Seattle to New York on June 5th"

검색 결과 (유사도 순):
1. enter_flight_locations (0.92) - 출발지/도착지 입력 워크플로우
2. select_date_for_travel (0.85) - 날짜 선택 워크플로우
3. select_oneway_trip (0.78) - 편도 선택 워크플로우
```

---

### 3.2 프롬프트 구성

워크플로우는 LLM 프롬프트의 **데모(exemplar)** 부분에 삽입됩니다.

#### 최종 프롬프트 구조

```
┌─────────────────────────────────────────────────────────────┐
│                      LLM 프롬프트 구조                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 1. System Message (역할 정의)                        │   │
│  │                                                      │   │
│  │ "You are a large language model trained to navigate │   │
│  │  the web. Output the next action and wait for the   │   │
│  │  next observation.                                   │   │
│  │                                                      │   │
│  │  Here is the action space:                          │   │
│  │  1. `CLICK [id]`: Click on an HTML element          │   │
│  │  2. `TYPE [id] [value]`: Type a string              │   │
│  │  3. `SELECT [id] [value]`: Select a value"          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 2. Demo Message (워크플로우 + 구체적 예제)            │   │
│  │    ↑ 워크플로우가 여기에 삽입됩니다!                   │   │
│  │                                                      │   │
│  │ ┌─────────────────────────────────────────────────┐ │   │
│  │ │ [워크플로우 텍스트]                              │ │   │
│  │ │                                                 │ │   │
│  │ │ ## enter_flight_locations                       │ │   │
│  │ │ Given that you are on the Delta flight booking  │ │   │
│  │ │ page, this workflow enters the departure and    │ │   │
│  │ │ destination city/airport.                       │ │   │
│  │ │ [link] From Departure Airport -> CLICK          │ │   │
│  │ │ [textbox] Origin City -> TYPE: {origin-city}    │ │   │
│  │ │ [link] {best-popup-option} -> CLICK             │ │   │
│  │ │ ...                                             │ │   │
│  │ │                                                 │ │   │
│  │ │ ## select_date_for_travel                       │ │   │
│  │ │ ...                                             │ │   │
│  │ └─────────────────────────────────────────────────┘ │   │
│  │                                                      │   │
│  │ ┌─────────────────────────────────────────────────┐ │   │
│  │ │ [구체적 예제들] (웹사이트별 필터링 + 랜덤 샘플링)  │ │   │
│  │ │                                                 │ │   │
│  │ │ Task: Find flights from LA to Chicago...        │ │   │
│  │ │ Observation: `<html>...</html>`                 │ │   │
│  │ │ Action: `CLICK [123]`                           │ │   │
│  │ │                                                 │ │   │
│  │ │ Task: Book a hotel in Miami...                  │ │   │
│  │ │ Observation: `<html>...</html>`                 │ │   │
│  │ │ Action: `TYPE [456] "Miami"`                    │ │   │
│  │ └─────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ 3. Query (현재 작업)                                 │   │
│  │                                                      │   │
│  │ Task: Find flights from Seattle to New York on      │   │
│  │       June 5th and only show those that can be      │   │
│  │       purchased with miles.                         │   │
│  │                                                      │   │
│  │ Trajectory:                                          │   │
│  │ Observation: `<previous HTML state>`                │   │
│  │ Action: `CLICK [789]`                               │   │
│  │                                                      │   │
│  │ Observation: `<current HTML state>`                 │   │
│  │ (다음 액션을 예측해야 함)                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Exemplar 로드 코드

```python
def get_exemplars(args) -> list:
    """워크플로우 + 구체적 예제를 결합하여 반환"""
    memory = []

    # 1. 워크플로우 메모리 로드 (추상화된 패턴)
    workflow_text = open(args.workflow_path, 'r').read().strip()
    if len(workflow_text):
        memory = [[{"role": "user", "content": workflow_text}]]

    # 2. 구체적 예제 로드
    with open(os.path.join(args.memory_path, "exemplars.json"), "r") as f:
        concrete_examples = json.load(f)

    # 3. 웹사이트별 필터링
    if any([args.website in cex[0].get("specifier", "") for cex in concrete_examples]):
        concrete_examples = [
            cex for cex in concrete_examples
            if all([tag in cex[0]["specifier"]
                   for tag in [args.domain, args.subdomain, args.website]])
        ]

    # 4. Top-K 랜덤 샘플링
    memory += random.sample(
        concrete_examples,
        min(args.retrieve_top_k, len(concrete_examples))
    )

    return memory
```

---

### 3.3 토큰 예산 관리

LLM의 컨텍스트 한도 내에서 최대한 많은 워크플로우/예제를 포함합니다.

#### 토큰 관리 로직

```python
# 최대 토큰 한도 정의
MAX_TOKENS = {
    "gpt-4": 8192,
    "gpt-4o": 128000,
    "gpt-3.5-turbo": 4096,
}

def eval_sample(task_id, args, sample):
    exemplars = get_exemplars(args)

    # 시스템 메시지 + 현재 쿼리의 토큰 수 계산
    total_num_tokens = num_tokens_from_messages(sys_message + query, args.model)

    # 토큰 한도 초과 시 건너뛰기
    if total_num_tokens > MAX_TOKENS[args.model]:
        logger.info(f"Too many tokens ({total_num_tokens}), skipping...")
        return

    # 데모 메시지 (워크플로우 + 예제) 구성
    demo_message = []
    for e_id, e in enumerate(exemplars):
        # 현재까지의 토큰 수 + 새 예제 토큰 수 계산
        total_num_tokens = num_tokens_from_messages(
            sys_message + demo_message + e + query, args.model
        )

        # 한도 초과 시 중단
        if total_num_tokens > MAX_TOKENS[args.model]:
            logger.info(f"Using {e_id} / {len(exemplars)} exemplars due to context limit")
            break
        else:
            demo_message.extend(e)  # 워크플로우/예제 추가

    # 최종 메시지 구성
    message = sys_message + demo_message + query
```

#### 토큰 예산 배분 전략

```
┌─────────────────────────────────────────────────────────────┐
│                    토큰 예산 배분 (예: 8192 토큰)             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────┐                   │
│  │ System Message                      │  ~200 토큰        │
│  │ (역할 정의 + 액션 스페이스)           │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
│  ┌─────────────────────────────────────┐                   │
│  │ 워크플로우 (고정)                    │  ~500-1000 토큰   │
│  │ 추상화된 패턴들                      │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
│  ┌─────────────────────────────────────┐                   │
│  │ 구체적 예제들 (가변)                 │  ~2000-4000 토큰  │
│  │ 토큰 한도 내에서 최대한 추가          │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
│  ┌─────────────────────────────────────┐                   │
│  │ 현재 쿼리 (고정)                     │  ~1000-3000 토큰  │
│  │ Task + Trajectory + Current HTML    │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
│  ┌─────────────────────────────────────┐                   │
│  │ 응답 버퍼                            │  ~500 토큰        │
│  │ LLM 출력을 위한 여유 공간            │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. 전체 파이프라인

### 오프라인 파이프라인

```
┌─────────────────────────────────────────────────────────────────────┐
│                        오프라인 전체 파이프라인                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Phase 1: 워크플로우 축적]                                          │
│                                                                     │
│  학습 데이터 (train/*.json)                                          │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────┐                                           │
│  │ offline_induction.py │                                           │
│  │ GPT-4o로 워크플로우 추출│                                          │
│  └──────────┬──────────┘                                           │
│             │                                                       │
│             ▼                                                       │
│  workflow/{website}.txt  ← 추상화된 워크플로우 저장                    │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  [Phase 2: 워크플로우 활용]                                          │
│                                                                     │
│  테스트 작업 + workflow/{website}.txt                                │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────┐                                           │
│  │   memory.py          │                                           │
│  │   get_exemplars()    │  워크플로우 + 구체적 예제 로드              │
│  └──────────┬──────────┘                                           │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                           │
│  │   프롬프트 구성       │                                           │
│  │   [Sys] + [Demo] +   │                                           │
│  │   [Query]            │                                           │
│  └──────────┬──────────┘                                           │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                           │
│  │   GPT 응답 생성      │  워크플로우 패턴 참고하여 액션 예측          │
│  └──────────┬──────────┘                                           │
│             │                                                       │
│             ▼                                                       │
│  results/{task_id}.json  ← 예측 결과 저장                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 온라인 파이프라인

```
┌─────────────────────────────────────────────────────────────────────┐
│                        온라인 전체 파이프라인                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [Iteration 1]                                                      │
│                                                                     │
│  테스트 작업 (워크플로우 없음)                                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────┐                                           │
│  │   에이전트 실행       │  일부 성공, 일부 실패                      │
│  └──────────┬──────────┘                                           │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                           │
│  │ online_induction.py  │                                           │
│  │ 성공 궤적에서 패턴 추출│                                           │
│  └──────────┬──────────┘                                           │
│             │                                                       │
│             ▼                                                       │
│  workflow/{website}.txt  ← 첫 번째 워크플로우 저장                    │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  [Iteration 2]                                                      │
│                                                                     │
│  테스트 작업 + workflow/{website}.txt                                │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────┐                                           │
│  │   에이전트 실행       │  더 많은 성공 (워크플로우 활용)             │
│  └──────────┬──────────┘                                           │
│             │                                                       │
│             ▼                                                       │
│  ┌─────────────────────┐                                           │
│  │ online_induction.py  │                                           │
│  │ 새로운 성공 패턴 추가  │                                           │
│  └──────────┬──────────┘                                           │
│             │                                                       │
│             ▼                                                       │
│  workflow/{website}.txt  ← 워크플로우 업데이트                        │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  [Iteration N]                                                      │
│                                                                     │
│       ...  반복하여 워크플로우 축적 및 성능 향상                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. 코드 참조

### Mind2Web 모듈

| 파일 | 역할 |
|------|------|
| `offline_induction.py` | 학습 데이터에서 워크플로우 유도 |
| `online_induction.py` | 에이전트 경험에서 워크플로우 유도 |
| `memory.py` | 워크플로우 로드 및 프롬프트 구성 |
| `workflow/retrieve.py` | FAISS 기반 의미론적 검색 |
| `utils/data.py` | 데이터 포맷팅 및 필터링 |
| `utils/llm.py` | LLM API 호출 및 토큰 관리 |
| `prompt/instruction_action.txt` | 워크플로우 유도 지시문 |
| `prompt/one_shot_action.txt` | 워크플로우 유도 예시 |

### WebArena 모듈

| 파일 | 역할 |
|------|------|
| `induce_rule.py` | 규칙 기반 워크플로우 유도 (중복 제거) |
| `induce_prompt.py` | 신경망 기반 워크플로우 유도 |
| `agents/legacy/dynamic_prompting.py` | 프롬프트 동적 구성 |
| `prompt/instruction.txt` | WebArena용 지시문 |
| `prompt/one_shot.txt` | WebArena용 예시 |

---

## 요약

| 단계 | 설명 | 핵심 코드 |
|------|------|----------|
| **워크플로우 축적** | 구체적 예제 → 추상화된 패턴 추출 | `offline_induction.py`, `online_induction.py` |
| **워크플로우 저장** | 웹사이트별 텍스트 파일로 저장 | `workflow/{website}.txt` |
| **워크플로우 검색** | 작업과 관련된 워크플로우 선택 | `workflow/retrieve.py` |
| **프롬프트 구성** | [System] + [Demo/Workflow] + [Query] | `memory.py` |
| **토큰 관리** | 컨텍스트 한도 내 최대 예제 포함 | `num_tokens_from_messages()` |
| **응답 생성** | 워크플로우 패턴 참고하여 액션 예측 | GPT-4o API 호출 |

워크플로우는 에이전트에게 **"이런 상황에서는 이런 패턴으로 행동하면 된다"**는 가이드를 제공하여, 새로운 작업에서도 효율적으로 행동할 수 있게 합니다.
