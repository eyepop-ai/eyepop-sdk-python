# EyePop Python SDK Architecture

## Overview

The EyePop Python SDK provides access to EyePop.ai's inference (Worker API) and data management (Data API) services. It is async-first (aiohttp) with transparent sync wrappers, supporting authentication, request tracing, load balancing, and visualization.

---

## High-Level Architecture

```mermaid
graph TB
    subgraph "User Application"
        UC[User Code]
    end

    subgraph "EyePop SDK"
        Factory["EyePopSdk<br/>(Factory)"]

        subgraph "Worker API"
            WE["WorkerEndpoint<br/>(async)"]
            SWE["SyncWorkerEndpoint<br/>(sync wrapper)"]
            WJ["WorkerJobs<br/>(Upload, LoadFrom)"]
            LB["EndpointLoadBalancer"]
            Pop["Pop / Components"]
        end

        subgraph "Data API"
            DE["DataEndpoint<br/>(async)"]
            SDE["SyncDataEndpoint<br/>(sync wrapper)"]
            DJ["DataJobs<br/>(Upload, Import, Infer)"]
            WS["WebSocket Events"]
        end

        subgraph "Shared Core"
            EP["Endpoint<br/>(base class)"]
            Jobs["Job<br/>(base class)"]
            Auth["Auth & Token Mgmt"]
            Retry["Retry Handlers"]
            Trace["RequestTracer"]
            Metrics["MetricCollector"]
            Settings["Settings<br/>(pydantic)"]
            Syncify["Syncify<br/>(event loop bridge)"]
        end

        subgraph "Data Types"
            PT["Prediction Types"]
            AT["Asset / Dataset Types"]
            MT["Model Types"]
            Arrow["Arrow Schemas"]
        end

        Viz["EyePopPlot<br/>(matplotlib)"]
    end

    subgraph "EyePop Services"
        WorkerSvc["Worker Service<br/>(inference pipelines)"]
        DataSvc["Data API Service<br/>(datasets, models)"]
        ComputeSvc["Compute API<br/>(session mgmt)"]
        AuthSvc["Auth Service"]
    end

    UC --> Factory
    Factory --> WE
    Factory --> SWE
    Factory --> DE
    Factory --> SDE
    Factory --> Viz

    SWE -.->|wraps| WE
    SDE -.->|wraps| DE

    WE --> EP
    DE --> EP
    WE --> WJ
    WE --> LB
    WE --> Pop
    DE --> DJ
    DE --> WS

    EP --> Auth
    EP --> Retry
    EP --> Trace
    EP --> Metrics
    EP --> Settings

    SWE --> Syncify
    SDE --> Syncify

    WJ --> Jobs
    DJ --> Jobs

    WJ --> PT
    DJ --> AT
    DJ --> MT
    DJ --> Arrow

    WE -->|HTTP/JSON-L| WorkerSvc
    DE -->|HTTP/JSON| DataSvc
    DE -->|WebSocket| DataSvc
    EP -->|Token exchange| AuthSvc
    WE -->|Session mgmt| ComputeSvc
```

---

## Class Hierarchy

```mermaid
classDiagram
    class ClientSession {
        <<interface>>
    }

    class Endpoint {
        -_session: aiohttp.ClientSession
        -_access_token: str
        -_token_expire_time: float
        +connect()
        +disconnect()
        +session_request()
        -_get_access_token()
        -_reconnect()*
    }

    class WorkerClientSession {
        <<interface>>
        +upload()
        +upload_stream()
        +load_from()
        +load_asset()
        +get_pop()
        +set_pop()
    }

    class WorkerEndpoint {
        -_pop_comp: Pop
        -_load_balancer: EndpointLoadBalancer
        -_pipeline_id: str
        +upload()
        +upload_stream()
        +load_from()
        +load_asset()
        +get_pop()
        +set_pop()
        -_reconnect()
    }

    class DataEndpoint {
        -_dataset_api_url: str
        -_vlm_api_url: str
        -_ws_connection: WebSocket
        +list_datasets()
        +create_dataset()
        +upload_asset()
        +import_asset()
        +list_models()
        +train_model()
        +infer()
        -_reconnect()
    }

    class SyncWorkerEndpoint {
        -_endpoint: WorkerEndpoint
        -_event_loop: asyncio.EventLoop
    }

    class SyncDataEndpoint {
        -_endpoint: DataEndpoint
        -_event_loop: asyncio.EventLoop
    }

    ClientSession <|.. Endpoint
    Endpoint <|-- WorkerEndpoint
    Endpoint <|-- DataEndpoint
    WorkerClientSession <|.. WorkerEndpoint
    WorkerEndpoint ..> SyncWorkerEndpoint : wrapped by
    DataEndpoint ..> SyncDataEndpoint : wrapped by
```

---

## Job Hierarchy

```mermaid
classDiagram
    class Job {
        -_queue: asyncio.Queue
        -_state: JobState
        -_callback: JobStateCallback
        +execute()
        +pop_result()
        +push_result()
        +cancel()
        #_do_execute_job()*
    }

    class WorkerJob {
        #_do_read_response()
        #_do_execute_job()
    }

    class UploadFileJob {
        -_location: str
        #_do_execute_job()
    }

    class UploadStreamJob {
        -_stream: BinaryIO
        -_mime_type: str
        #_do_execute_job()
    }

    class LoadFromJob {
        -_location: str
        #_do_execute_job()
    }

    class LoadFromAssetUuidJob {
        -_asset_uuid: str
        #_do_execute_job()
    }

    class DataJob {
        #_do_execute_job()
        +result()
    }

    class DataUploadStreamJob {
        -_stream: BinaryIO
    }

    class ImportFromJob {
        -_asset_import: AssetImport
    }

    class InferJob {
        -_request: InferRequest
    }

    class EvaluateJob {
        -_request: EvaluateRequest
    }

    Job <|-- WorkerJob
    Job <|-- DataJob
    WorkerJob <|-- UploadFileJob
    WorkerJob <|-- UploadStreamJob
    WorkerJob <|-- LoadFromJob
    WorkerJob <|-- LoadFromAssetUuidJob
    DataJob <|-- DataUploadStreamJob
    DataJob <|-- ImportFromJob
    DataJob <|-- InferJob
    DataJob <|-- EvaluateJob
```

---

## Job State Machine

```mermaid
stateDiagram-v2
    [*] --> CREATED
    CREATED --> STARTED : execute()
    STARTED --> IN_PROGRESS : first result
    IN_PROGRESS --> IN_PROGRESS : more results
    IN_PROGRESS --> FINISHED : stream ends
    STARTED --> FINISHED : empty result
    STARTED --> FAILED : error
    IN_PROGRESS --> FAILED : error
    FINISHED --> DRAINED : all results consumed
    FAILED --> DRAINED : error consumed
```

---

## Authentication Flow

```mermaid
sequenceDiagram
    participant User
    participant SDK as EyePopSdk
    participant EP as Endpoint
    participant Auth as Auth Service
    participant Compute as Compute API

    User->>SDK: workerEndpoint(api_key=...)
    SDK->>EP: create WorkerEndpoint

    alt Has explicit access_token
        EP->>EP: Use token directly
    else Has secret_key (named Pop)
        EP->>Auth: POST /authentication/token<br/>{secret_key}
        Auth-->>EP: {access_token, expires_at}
    else Has api_key (transient)
        EP->>Compute: POST /v1/auth/authenticate<br/>{api_key}
        Compute-->>EP: {access_token, expires_at}
    end

    EP->>EP: Cache token + expiry

    Note over EP: On subsequent requests
    EP->>EP: Check token expiry (-60s buffer)
    alt Token expired
        EP->>Auth: Refresh token
        Auth-->>EP: New token
    end

    Note over EP: On 401 response
    EP->>Auth: Force refresh token
    Auth-->>EP: New token
    EP->>EP: Retry original request
```

---

## Worker API: Inference Flow

```mermaid
sequenceDiagram
    participant User
    participant WE as WorkerEndpoint
    participant LB as LoadBalancer
    participant Worker as Worker Service
    participant Compute as Compute API

    User->>WE: async with endpoint
    WE->>WE: connect() → create aiohttp session

    alt Transient Pipeline
        WE->>Compute: GET /v1/auth/sessions
        Compute-->>WE: Session config (endpoints)
        WE->>Worker: POST /pipelines
        Worker-->>WE: pipeline_id
    else Named Pop
        WE->>Worker: GET /pops/{pop_id}/config
        Worker-->>WE: Pop config + endpoints
    end

    WE->>LB: Initialize with endpoints

    User->>WE: upload("image.jpg")
    WE->>WE: Create UploadFileJob
    WE->>WE: Start async task

    Note over WE,Worker: Image Upload (sync mode)
    WE->>LB: Get healthy endpoint
    LB-->>WE: endpoint_url
    WE->>Worker: PATCH /pipelines/{id}/source<br/>mode=queue, processing=sync
    Worker-->>WE: JSON-L prediction stream

    WE->>WE: Parse lines → push to queue

    User->>WE: job.predict()
    WE-->>User: Prediction result dict

    Note over WE,Worker: Video Upload (full-duplex)
    User->>WE: upload("video.mp4")
    WE->>Worker: POST /pipelines/{id}/prepareSource
    Worker-->>WE: source_id

    par Upload stream
        WE->>Worker: POST /source?sourceId=...
    and Read predictions
        Worker-->>WE: JSON-L prediction stream
    end

    loop For each frame
        User->>WE: job.predict()
        WE-->>User: Frame prediction
    end

    User->>WE: disconnect()
    alt Transient Pipeline
        WE->>Worker: DELETE /pipelines/{id}
    end
```

---

## Data API: Dataset & Model Flow

```mermaid
sequenceDiagram
    participant User
    participant DE as DataEndpoint
    participant DataSvc as Data API
    participant VLM as VLM API

    User->>DE: async with endpoint
    DE->>DataSvc: GET /configs
    DataSvc-->>DE: {dataset_api_url, vlm_api_url}

    opt WebSocket enabled
        DE->>DataSvc: WS /events
        DataSvc-->>DE: Connected
        DE->>DataSvc: Subscribe(account_uuid)
    end

    Note over User,DataSvc: Dataset Operations
    User->>DE: list_datasets()
    DE->>DataSvc: GET /datasets?account_uuid=...
    DataSvc-->>DE: [Dataset, ...]
    DE-->>User: list[Dataset]

    User->>DE: create_dataset(DatasetCreate)
    DE->>DataSvc: POST /datasets
    DataSvc-->>DE: Dataset
    DE-->>User: Dataset

    Note over User,DataSvc: Asset Upload
    User->>DE: upload_asset(stream, mime_type)
    DE->>DataSvc: POST /assets?dataset_uuid=...
    DataSvc-->>DE: Asset
    DE-->>User: Asset

    Note over User,VLM: VLM Inference
    User->>DE: infer(request)
    DE->>VLM: POST /infer
    VLM-->>DE: Chunked response
    DE-->>User: InferResponse

    Note over User,DataSvc: Model Training
    User->>DE: train_model(model_uuid)
    DE->>DataSvc: POST /models/{id}/train
    DataSvc-->>DE: Training progress stream
    DE-->>User: ModelTrainingProgress

    opt WebSocket Events
        DataSvc->>DE: ChangeEvent (asset_status_modified)
        DE->>User: EventHandler callback
    end
```

---

## Pop Component Pipeline

```mermaid
graph LR
    subgraph "Pop Configuration"
        direction TB
        Input["Input Source<br/>(image/video)"]

        subgraph "Component Chain"
            C1["InferenceComponent<br/>(e.g. object detection)"]
            C2["TrackingComponent<br/>(multi-object tracker)"]
            C3["ForwardComponent<br/>(crop detected objects)"]
            C4["InferenceComponent<br/>(e.g. classification)"]
            C5["ContourFinderComponent<br/>(contour extraction)"]
        end

        Output["Prediction Results<br/>(JSON-L stream)"]
    end

    Input --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C1 --> C5
    C4 --> Output
    C5 --> Output

    style C1 fill:#4a90d9,color:#fff
    style C2 fill:#7b68ee,color:#fff
    style C3 fill:#e67e22,color:#fff
    style C4 fill:#4a90d9,color:#fff
    style C5 fill:#27ae60,color:#fff
```

---

## Sync/Async Bridge

```mermaid
graph TB
    subgraph "User Thread (main)"
        SyncCall["sync_endpoint.upload()"]
        SyncResult["Return result"]
    end

    subgraph "Syncify Layer"
        Bridge["run_coroutine_threadsafe()"]
        Future["concurrent.futures.Future"]
    end

    subgraph "Daemon Thread (event loop)"
        Loop["asyncio.EventLoop"]
        AsyncCall["endpoint.upload()"]
        AsyncJob["WorkerJob.execute()"]
        HTTP["aiohttp request"]
    end

    SyncCall --> Bridge
    Bridge --> Loop
    Loop --> AsyncCall
    AsyncCall --> AsyncJob
    AsyncJob --> HTTP
    HTTP --> AsyncJob
    AsyncJob --> Future
    Future --> SyncResult

    style Bridge fill:#f39c12,color:#fff
    style Loop fill:#3498db,color:#fff
```

---

## Load Balancer & Retry

```mermaid
flowchart TB
    Request["HTTP Request"] --> LB["EndpointLoadBalancer"]
    LB --> Select["Select healthy endpoint<br/>(round-robin)"]
    Select --> Send["Send request"]

    Send --> Status{Response?}

    Status -->|200 OK| Success["Return response"]
    Status -->|401| Refresh["Refresh auth token"]
    Refresh --> Retry1["Retry request"]
    Retry1 --> Send

    Status -->|404| ReConfig["Refresh endpoint config"]
    ReConfig --> Send

    Status -->|500/502/503| Backoff["Exponential backoff<br/>2^(attempt-1) seconds"]
    Backoff --> MaxRetry{Max retry<br/>time?}
    MaxRetry -->|No| Send
    MaxRetry -->|Yes| Fail["Raise exception"]

    Status -->|Connection Error| Mark["Mark endpoint unhealthy"]
    Mark --> AllDown{All endpoints<br/>unhealthy?}
    AllDown -->|No| Select
    AllDown -->|Yes| ReConfig
```

---

## Module Dependency Map

```mermaid
graph TB
    subgraph "Public API"
        init["__init__.py"]
        sdk["eyepopsdk.py"]
    end

    subgraph "Core"
        ep["endpoint.py"]
        jobs["jobs.py"]
        cs["client_session.py"]
        sync["syncify.py"]
        settings["settings.py"]
        exc["exceptions.py"]
    end

    subgraph "Worker"
        we["worker_endpoint.py"]
        wj["worker_jobs.py"]
        wt["worker_types.py"]
        ws["worker_syncify.py"]
        lb["load_balancer.py"]
    end

    subgraph "Data"
        de["data_endpoint.py"]
        dj["data_jobs.py"]
        dt["data/types/*"]
        ds["data_syncify.py"]
        da["data/arrow/*"]
    end

    subgraph "Infra"
        trace["request_tracer.py"]
        metrics["metrics.py"]
        periodic["periodic.py"]
        compute["compute/*"]
    end

    subgraph "Viz"
        viz["visualize.py"]
    end

    init --> sdk
    init --> viz
    sdk --> we
    sdk --> de
    sdk --> settings

    we --> ep
    we --> wj
    we --> lb
    we --> wt
    we --> compute
    ws --> we
    ws --> sync

    de --> ep
    de --> dj
    de --> dt
    ds --> de
    ds --> sync

    wj --> jobs
    dj --> jobs

    ep --> cs
    ep --> trace
    ep --> metrics
    ep --> periodic
    ep --> settings
    ep --> exc

    style init fill:#e74c3c,color:#fff
    style sdk fill:#e74c3c,color:#fff
    style ep fill:#3498db,color:#fff
    style we fill:#2ecc71,color:#fff
    style de fill:#9b59b6,color:#fff
```

---

## Error Handling Hierarchy

```mermaid
graph TB
    Base["Exception"]
    Pop["PopNotStartedException"]
    Config["PopConfigurationException"]
    Reach["PopNotReachableException"]
    Session["ComputeSessionException"]
    Token["ComputeTokenException"]
    Health["ComputeHealthCheckException"]

    Base --> Pop
    Base --> Config
    Base --> Reach
    Base --> Session
    Base --> Token
    Base --> Health

    Pop ---|"Pop not started<br/>with auto_start=False"| P1[" "]
    Config ---|"Invalid Pop<br/>configuration"| P2[" "]
    Reach ---|"No healthy<br/>endpoints"| P3[" "]
    Session ---|"Session create<br/>or manage failed"| P4[" "]
    Token ---|"Token exchange<br/>failed"| P5[" "]
    Health ---|"Health check<br/>timeout"| P6[" "]

    style P1 fill:none,stroke:none
    style P2 fill:none,stroke:none
    style P3 fill:none,stroke:none
    style P4 fill:none,stroke:none
    style P5 fill:none,stroke:none
    style P6 fill:none,stroke:none
```

---

## Key Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| **Factory** | `EyePopSdk.workerEndpoint()` / `dataEndpoint()` | Resolves auth, creates appropriate endpoint type |
| **Template Method** | `Job.execute()` → `_do_execute_job()` | Job lifecycle with subclass-specific execution |
| **Adapter** | `SyncWorkerEndpoint` / `SyncDataEndpoint` | Async-to-sync interface conversion |
| **Strategy** | Retry handlers per HTTP status | Pluggable error recovery strategies |
| **Observer** | WebSocket `EventHandler` callbacks | Real-time dataset/model change notifications |
| **Builder** | Pop component `forward` chaining | Composable inference pipeline configuration |
| **Semaphore** | `asyncio.Semaphore(job_queue_length)` | Backpressure on concurrent job submission |
| **Round-Robin** | `EndpointLoadBalancer` | Distribute requests across healthy endpoints |
