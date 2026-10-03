# Black Box — AI Agent Flight Recorder

> **Black Box** is an AI-powered debugging system that records, diagnoses, and replays agent execution traces to pinpoint failure-causing steps across any LLM provider.

---

## Architecture Overview

```
Your Agent
    │
    ▼
┌─────────────────────────────┐
│  Black Box Proxy  :8000     │  ◄── Universal Intercept Layer
│  (FastAPI + httpx)          │
└────────────┬────────────────┘
             │  Records every call
             ▼
┌─────────────────────────────┐
│  SQLite Trace Store         │  ◄── blackbox_traces.db
│  (SQLAlchemy async)         │
└────────────┬────────────────┘
             │
             ├── feature/trace-engine     (Step Reconstruction + Replay)
             ├── feature/fault-injector   (Synthetic Failure Dataset)
             ├── feature/diagnosis-model  (ML Failure Localization)
             └── feature/dashboard-ui    (Web Dashboard + Diff Viewer)
```

## Supported LLM Providers
| Provider | Route Prefix | Example |
|---|---|---|
| OpenAI / Groq / Ollama | `/v1/...` | `/v1/chat/completions` |
| Google Gemini | `/google/...` | `/google/v1beta/models/...` |
| Anthropic Claude | `/anthropic/...` | `/anthropic/v1/messages` |
| Custom / Generic | `/proxy/...` | any upstream base URL |

## Quick Start

```bash
pip install -r requirements.txt
uvicorn blackbox.proxy.server:app --port 8000 --reload
```

Then point your agent at `http://localhost:8000` instead of the real provider base URL, and add the header:
```
X-BlackBox-Run-ID: my-agent-run-001
```

## Project Structure
```
blackbox/
  proxy/        ← Proxy server, streaming, replay engine
  traces/       ← DB models, schemas, recorder, reconstructor
  faults/       ← Fault injection engine
  model/        ← Feature extraction, ML diagnosis model
  verify/       ← Counterfactual verifier
  cli/          ← Typer CLI commands
  web/          ← Dashboard API endpoints
agents/         ← Test agent scripts
experiments/    ← Dataset generation, evaluation scripts
tests/          ← pytest test suites
```

## Running Tests
```bash
PYTHONPATH=. pytest tests/ -v
```
