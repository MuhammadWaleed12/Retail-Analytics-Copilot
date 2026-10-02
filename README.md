# Retail Analytics Copilot (DSPy + LangGraph)

A local, free AI agent that answers retail analytics questions by combining RAG over local docs and SQL over a local SQLite database (Northwind).

[![CI](https://github.com/MuhammadWaleed12/Retail-Analytics-Copilot/actions/workflows/ci.yml/badge.svg)](https://github.com/MuhammadWaleed12/Retail-Analytics-Copilot/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Local-first](https://img.shields.io/badge/LLM-local--first-2ea44f)](#local-first-design)

## Why this project

Retail questions often span two different knowledge sources: structured facts in a database and business rules in documents. This project demonstrates an inspectable, local-first workflow that routes each question to SQL, retrieval, or both, and returns the executed SQL, citations, confidence, and a short explanation alongside the answer.

## Architecture

```mermaid
flowchart LR
    Q[Question] --> R{DSPy router}
    R -->|rag| D[TF-IDF document retrieval]
    R -->|sql| N[NL-to-SQL]
    R -->|hybrid| D
    D --> P[Constraint planner]
    P --> N
    N --> E[SQLite executor]
    E -->|SQL error or invalid output| X[Repair loop]
    X -->|maximum 2 retries| N
    D --> S[Answer synthesizer]
    E --> S
    S --> O[Typed answer + citations + confidence]
```

The repository is intentionally small enough to audit end to end: documents and the sample database stay local, retrieval uses TF-IDF, and the default model runs through Ollama.

## Project Structure

```
assessment/
├── agent/
│   ├── graph_hybrid.py          # LangGraph agent with 6+ nodes + repair loop
│   ├── dspy_signatures.py       # DSPy modules (Router, NL-SQL, Synthesizer)
│   ├── rag/
│   │   └── retrieval.py         # TF-IDF retriever
│   └── tools/
│       └── sqlite_tool.py        # SQLite DB access + schema introspection
├── data/
│   └── northwind.sqlite         # Northwind sample database
├── docs/
│   ├── marketing_calendar.md    # Marketing campaign dates
│   ├── kpi_definitions.md       # KPI formulas (AOV, Gross Margin)
│   ├── catalog.md               # Product categories
│   └── product_policy.md        # Return policies
├── sample_questions_hybrid_eval.jsonl  # Evaluation questions
├── run_agent_hybrid.py          # Main CLI entrypoint
└── requirements.txt
```

## Graph Design

The LangGraph agent implements a hybrid RAG + SQL architecture with the following nodes:

1. **Router** (DSPy): Classifies questions as `rag`, `sql`, or `hybrid`
2. **Retriever**: Retrieves top-k document chunks using TF-IDF
3. **Planner**: Extracts constraints (date ranges, categories, KPI formulas) from question and docs
4. **NL→SQL** (DSPy): Generates SQLite queries using schema introspection
5. **Executor**: Executes SQL and captures results/errors
6. **Synthesizer** (DSPy): Produces typed answers matching format_hint with citations
7. **Repair Loop**: Up to 2 repair iterations on SQL errors or invalid outputs
8. **Confidence Calculator**: Computes confidence based on retrieval scores, SQL success, and repair count

The graph routes questions based on type:
- **rag**: Document-only questions → Retriever → Synthesizer
- **sql**: Database-only questions → NL-to-SQL → Executor → Synthesizer
- **hybrid**: Questions requiring both → Retriever → Planner → NL-to-SQL → Executor → Synthesizer

## Evaluation and optimization status

`optimize_dspy.py` is an experimental BootstrapFewShot script, not a reproducible
benchmark. It contains **three training examples** and uses those same three
examples for its before/after evaluation. Its metric counts queries that execute
without an error; it does not compare returned values with expected answers.
A query can execute successfully and still answer the wrong question.

The script also needs integration work before its results can be relied on:
`NLToSQL.forward()` returns a DSPy prediction with a `sql_query` field, while
`evaluate_sql_module()` currently passes that prediction directly to string
cleanup. The script does not save its compiled module, and the CLI does not load
an optimized artifact. Running the CLI therefore does not demonstrate use of a
trained optimizer.

The previously stated 60% → 85% accuracy improvement and 200 ms latency cost are
not established by a reproducible benchmark in this repository and have been
removed. No model-quality or latency improvement is claimed here.

### Inspect a local agent run

After completing setup and starting Ollama, run the six bundled questions into a
fresh output file:

```bash
python run_agent_hybrid.py \
    --batch sample_questions_hybrid_eval.jsonl \
    --out local-results.jsonl
```

Inspect each result's answer, SQL, citations, and explanation against the source
database and documents. The six questions are smoke-test inputs: they contain
format hints but no expected answers, so processing all six is not an accuracy
score. The committed `outputs_hybrid.jsonl` is an example output, not evidence
that the current code and model passed a fresh evaluation.

For a meaningful before/after comparison:

1. Fix and validate the optimizer integration, then save and explicitly load the
   compiled module for the optimized run.
2. Keep training questions separate from held-out evaluation questions and add
   independently checked expected answers to the latter.
3. Record the Git commit, dependency versions, model tag/digest and settings,
   dataset version, hardware, and whether latency includes model warm-up.
4. Report answer correctness separately from SQL execution success, retrieval
   relevance, and citation support. Count errors rather than dropping them.
5. Run both configurations on the same held-out inputs and publish the per-case
   results, denominators, and latency measurements.

The fast unit suite below checks deterministic tool behavior. It does not measure
LLM answer quality or optimizer performance.

## Assumptions & Trade-offs

1. **CostOfGoods Approximation**: Uses 70% of UnitPrice when cost data is unavailable (as specified in assignment)
2. **Chunking Strategy**: Documents are chunked at paragraph level (double newline split)
3. **Retrieval**: TF-IDF with cosine similarity (no external dependencies beyond scikit-learn)
4. **Model**: Uses Phi-3.5-mini-instruct via Ollama (local, free)
5. **Repair Limit**: Maximum 2 repair iterations to prevent infinite loops
6. **Date Extraction**: Marketing calendar dates are hardcoded in planner (1997-06-01 to 1997-06-30 for Summer Beverages, 1997-12-01 to 1997-12-31 for Winter Classics)

## Setup

### Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/) for full agent runs
- `sqlite3` and `curl` only if you want to rebuild the included sample database

### Quick start

1. Clone the repository and create a virtual environment:

```bash
git clone https://github.com/MuhammadWaleed12/Retail-Analytics-Copilot.git
cd Retail-Analytics-Copilot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Pull the default local model:

```bash
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M
```

The Northwind sample database is already included. To rebuild it from the upstream dataset:

```bash
mkdir -p data
curl -L -o data/northwind.sqlite https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

# Create compatibility views
sqlite3 data/northwind.sqlite <<'SQL'
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM Orders;
CREATE VIEW IF NOT EXISTS order_items AS SELECT * FROM "Order Details";
CREATE VIEW IF NOT EXISTS products AS SELECT * FROM Products;
CREATE VIEW IF NOT EXISTS customers AS SELECT * FROM Customers;
SQL
```

## Usage

Make sure Ollama is running, then process the included evaluation questions:

```bash
python run_agent_hybrid.py \
    --batch sample_questions_hybrid_eval.jsonl \
    --out outputs_hybrid.jsonl
```

## Output Format

Each line in `outputs_hybrid.jsonl` follows this structure:

```json
{
    "id": "question_id",
    "final_answer": <matches format_hint>,
    "sql": "<last executed SQL or empty if RAG-only>",
    "confidence": 0.0-1.0,
    "explanation": "<= 2 sentences>",
    "citations": ["Orders", "Products", "kpi_definitions::chunk0", ...]
}
```

## Testing

Run the fast unit suite (no Ollama service required):

```bash
python -m unittest discover -s tests -v
```

The unit tests cover SQLite schema discovery, query execution, read-only mutation rejection, and errors, plus document loading, ranking, unknown terms, and empty-document behavior. GitHub Actions runs them on Python 3.10, 3.11, and 3.12.

The evaluation file contains six end-to-end questions covering:

- RAG-only policy questions
- SQL-only analytics such as top products by revenue
- Hybrid questions that combine campaign dates or KPI definitions with SQL

## Local-first design

- No hosted vector database is required; retrieval runs over files in `docs/`.
- No hosted LLM API is required; the default model is served locally by Ollama.
- Answers expose their SQL and citations so results can be inspected instead of treated as a black box.

## Limitations

- TF-IDF is lexical retrieval and will miss some semantic matches.
- Retrieval returns at most `top_k` chunks with positive lexical similarity;
  empty or unknown-term queries return no chunks. Unmatched documents are not
  added merely to fill the result count. A positive score indicates term overlap,
  not proof that a document supports the answer.
- The bundled database is sample data, not a production retail warehouse.
- Confidence is a workflow heuristic based on retrieval, SQL success, and repairs; it is not a calibrated probability.
- SQLite connections enable `PRAGMA query_only`, so generated SQL cannot mutate the source database.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development and pull-request workflow.
