"""LangGraph hybrid agent for retail analytics."""
import json
import re
from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

import dspy
from agent.dspy_signatures import Router, NLToSQL, Synthesizer
from agent.tools.sqlite_tool import SQLiteTool
from agent.rag.retrieval import TFIDFRetriever


class AgentState(TypedDict):
    """State for the hybrid agent."""
    question: str
    format_hint: str
    route: str
    retrieved_chunks: list
    constraints: dict
    sql_query: str
    sql_results: list
    sql_columns: list
    sql_error: str
    final_answer: str
    citations: list
    explanation: str
    confidence: float
    repair_count: int
    trace: list


def create_hybrid_agent(db_path: str, docs_dir: str, model_name: str = "phi3.5:3.8b-mini-instruct-q4_K_M"):
    """Create and configure the hybrid agent graph."""
    
    # Initialize DSPy with Ollama
    # DSPy 3.0+ supports Ollama via ollama_chat/ prefix
    try:
        # Try using ollama_chat/ prefix (DSPy 3.0+)
        ollama_model = f"ollama_chat/{model_name}"
        lm = dspy.LM(
            model=ollama_model, 
            api_base="http://localhost:11434", 
            api_key=""
        )
        dspy.configure(lm=lm)
    except Exception as e1:
        try:
            # Fallback: Try langchain_ollama integration
            from langchain_ollama import OllamaLLM
            from dspy.langchain import LangChainLM
            
            ollama_llm = OllamaLLM(model=model_name)
            lm = LangChainLM(ollama_llm)
            dspy.configure(lm=lm)
        except Exception as e2:
            # Last fallback: Try direct model name
            print(f"Warning: Could not configure Ollama via ollama_chat or langchain. Trying direct model name.")
            print(f"Errors: {e1}, {e2}")
            lm = dspy.LM(model=model_name)
            dspy.configure(lm=lm)
    
    # Initialize tools
    sql_tool = SQLiteTool(db_path)
    retriever = TFIDFRetriever(docs_dir)
    
    # Initialize DSPy modules
    router = Router()
    nl_to_sql = NLToSQL()
    synthesizer = Synthesizer()
    
    # Get schema once
    schema = sql_tool.get_schema()
    
    def router_node(state: AgentState) -> AgentState:
        """Route question to rag, sql, or hybrid."""
        question = state["question"]
        
        # Force routing based on keywords if router fails
        question_lower = question.lower()
        route = None
        
        # Check for SQL indicators
        sql_keywords = ["top", "revenue", "total", "sum", "average", "count", "margin", "quantity", "sold"]
        hybrid_keywords = ["during", "campaign", "summer", "winter", "1997", "aov", "gross margin"]
        
        if any(kw in question_lower for kw in sql_keywords) or any(kw in question_lower for kw in hybrid_keywords):
            if any(kw in question_lower for kw in ["policy", "return", "definition", "what is"]):
                route = "hybrid"
            elif any(kw in question_lower for kw in hybrid_keywords):
                route = "hybrid"
            else:
                route = "sql"
        
        # If no keywords match, try router
        if route is None:
            try:
                result = router(question=question)
                route = result.route if hasattr(result, 'route') else str(result)
            except Exception as e:
                state["trace"].append(f"Router error: {e}, defaulting to hybrid")
                route = "hybrid"  # Default to hybrid for safety
        
        state["route"] = route
        state["trace"].append(f"Router: {route} (question: {question[:50]}...)")
        return state
    
    def retriever_node(state: AgentState) -> AgentState:
        """Retrieve relevant document chunks."""
        question = state["question"]
        chunks = retriever.retrieve(question, top_k=5)
        
        state["retrieved_chunks"] = [
            {
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "source": chunk.source,
                "score": chunk.score
            }
            for chunk in chunks
        ]
        state["trace"].append(f"Retrieved {len(chunks)} chunks")
        return state
    
    def planner_node(state: AgentState) -> AgentState:
        """Extract constraints from question and retrieved docs."""
        question = state["question"]
        chunks = state.get("retrieved_chunks", [])
        
        constraints = {
            "date_ranges": [],
            "categories": [],
            "kpi_formula": None,
            "entities": []
        }
        
        # Extract date ranges from marketing calendar
        for chunk in chunks:
            if "marketing_calendar" in chunk["source"]:
                content = chunk["content"]
                # Look for date patterns
                date_pattern = r'(\d{4}-\d{2}-\d{2})'
                dates = re.findall(date_pattern, content)
                if dates:
                    constraints["date_ranges"].extend(dates)
                
                # Look for campaign names
                if "Summer Beverages" in content:
                    constraints["date_ranges"].append(("1997-06-01", "1997-06-30"))
                elif "Winter Classics" in content:
                    constraints["date_ranges"].append(("1997-12-01", "1997-12-31"))
        
        # Extract KPI formulas
        for chunk in chunks:
            if "kpi_definitions" in chunk["source"]:
                content = chunk["content"]
                if "AOV" in content or "Average Order Value" in content:
                    constraints["kpi_formula"] = "AOV"
                elif "Gross Margin" in content or "GM" in content:
                    constraints["kpi_formula"] = "GM"
        
        # Extract categories
        category_keywords = ["Beverages", "Condiments", "Confections", "Dairy Products", 
                           "Grains/Cereals", "Meat/Poultry", "Produce", "Seafood"]
        for keyword in category_keywords:
            if keyword.lower() in question.lower():
                constraints["categories"].append(keyword)
        
        state["constraints"] = constraints
        state["trace"].append(f"Constraints: {constraints}")
        return state
    
    def nl_to_sql_node(state: AgentState) -> AgentState:
        """Generate SQL query from natural language."""
        question = state["question"]
        constraints = state.get("constraints", {})
        
        # Format constraints string
        constraints_str = json.dumps(constraints, indent=2)
        
        try:
            result = nl_to_sql(
                question=question,
                schema=schema,
                constraints=constraints_str
            )
            # Extract SQL query from result
            if hasattr(result, 'sql_query'):
                sql_query = result.sql_query
            elif hasattr(result, 'query'):
                sql_query = result.query
            else:
                sql_query = str(result)

            # Ensure we got a string
            sql_query = str(sql_query).strip()

            # Debug: log raw SQL before cleaning
            state["trace"].append(f"RAW SQL from LLM (first 200 chars): {sql_query[:200]}")
        except Exception as e:
            state["trace"].append(f"Error generating SQL: {str(e)}")
            sql_query = ""

        # Clean SQL query (remove markdown code blocks if present)
        sql_query = re.sub(r'```sql\n?', '', sql_query)
        sql_query = re.sub(r'```\n?', '', sql_query)
        sql_query = sql_query.strip()

        # Remove DSPy output artifacts (like :0, :1, etc. at line ends)
        sql_query = re.sub(r':\d+\s*\n', '\n', sql_query)  # Remove :0, :1, etc. at line ends
        sql_query = re.sub(r':\d+\s*$', '', sql_query)  # Remove :0, :1, etc. at end of string
        sql_query = re.sub(r':\d+(\s+|,|\))', r'\1', sql_query)  # Remove :0, :1, etc. followed by space/comma/paren

        # Fix common table name issues
        sql_query = sql_query.replace('OrderDetails', '"Order Details"')
        sql_query = sql_query.replace('orderdetails', '"Order Details"')
        sql_query = sql_query.replace('Order_Details', '"Order Details"')

        # Fix common SQL syntax errors
        # Fix truncated BETWEEN (but avoid creating BETWEENEN)
        sql_query = re.sub(r'\bBETWEENEN\b', 'BETWEEN', sql_query)  # Fix double-fix first
        sql_query = re.sub(r'\bBETWE\b', 'BETWEEN', sql_query)  # Fix truncated BETWEEN with word boundary
        # Remove triple/double quotes first
        sql_query = re.sub(r"'''(\d{4}-\d{2}-\d{2})'''", r"'\1'", sql_query)
        sql_query = re.sub(r"''(\d{4}-\d{2}-\d{2})''", r"'\1'", sql_query)
        # Quote date strings (only if not already quoted)
        sql_query = re.sub(r"(?<!')(\d{4}-\d{2}-\d{2})(?!')", r"'\1'", sql_query)

        # Check for incomplete SQL (must end with ; or have proper structure)
        sql_query = sql_query.rstrip()
        if sql_query and not sql_query.endswith(';'):
            # Add semicolon if missing (helps SQLite parse complete statement)
            sql_query = sql_query + ';'

        state["sql_query"] = sql_query
        if sql_query:
            state["trace"].append(f"Generated SQL: {sql_query[:100]}...")
            state["trace"].append(f"Generated SQL (char count): {len(sql_query)}")
        else:
            state["trace"].append("Warning: Generated SQL query is empty")
        return state
    
    def executor_node(state: AgentState) -> AgentState:
        """Execute SQL query."""
        sql_query = state.get("sql_query", "").strip()
        
        if not sql_query:
            state["sql_error"] = "No SQL query to execute"
            state["trace"].append("Executor: No SQL query provided")
            return state
        
        # Clean SQL query again (remove markdown if still present)
        sql_query = re.sub(r'```sql\n?', '', sql_query)
        sql_query = re.sub(r'```\n?', '', sql_query)
        sql_query = sql_query.strip()

        # Remove DSPy output artifacts (like :0, :1, etc.)
        sql_query = re.sub(r':\d+\s*\n', '\n', sql_query)
        sql_query = re.sub(r':\d+\s*$', '', sql_query)
        sql_query = re.sub(r':\d+(\s+|,|\))', r'\1', sql_query)

        # Fix common SQL syntax errors before execution
        sql_query = re.sub(r'\bBETWEENEN\b', 'BETWEEN', sql_query)  # Fix double-fix first
        sql_query = re.sub(r'\bBETWE\b', 'BETWEEN', sql_query)  # Fix truncated BETWEEN
        sql_query = sql_query.replace('OrderDetails', '"Order Details"')
        sql_query = sql_query.replace('orderdetails', '"Order Details"')
        # Remove triple/double quotes first
        sql_query = re.sub(r"'''(\d{4}-\d{2}-\d{2})'''", r"'\1'", sql_query)
        sql_query = re.sub(r"''(\d{4}-\d{2}-\d{2})''", r"'\1'", sql_query)
        # Quote date strings (only if not already quoted)
        sql_query = re.sub(r"(?<!')(\d{4}-\d{2}-\d{2})(?!')", r"'\1'", sql_query)
        
        state["sql_query"] = sql_query  # Update cleaned query

        # Debug: log full SQL (write to trace for debugging)
        state["trace"].append(f"Executing SQL (full): {sql_query}")
        state["trace"].append(f"Executing SQL preview: {sql_query[:100]}...")

        # Validate SQL has proper structure
        if sql_query:
            # Check for unclosed quotes
            single_quote_count = sql_query.count("'") - sql_query.count("\\'")
            double_quote_count = sql_query.count('"') - sql_query.count('\\"')

            if single_quote_count % 2 != 0:
                state["sql_error"] = "SQL has unclosed single quotes"
                state["trace"].append(f"SQL validation error: unclosed single quotes")
                return state

            if double_quote_count % 2 != 0:
                state["sql_error"] = "SQL has unclosed double quotes"
                state["trace"].append(f"SQL validation error: unclosed double quotes")
                return state
        
        rows, columns, error = sql_tool.execute_query(sql_query)
        
        if error:
            state["sql_error"] = error
            state["sql_results"] = []
            state["sql_columns"] = []
            state["trace"].append(f"SQL error: {error}")
        else:
            state["sql_error"] = ""
            state["sql_results"] = rows
            state["sql_columns"] = columns
            state["trace"].append(f"SQL executed successfully: {len(rows)} rows returned")
        
        return state
    
    def synthesizer_node(state: AgentState) -> AgentState:
        """Synthesize final answer."""
        question = state["question"]
        format_hint = state["format_hint"]
        sql_results = state.get("sql_results", [])
        sql_columns = state.get("sql_columns", [])
        sql_query = state.get("sql_query", "")
        chunks = state.get("retrieved_chunks", [])
        route = state.get("route", "")
        
        # Format SQL results with column names
        route = state.get("route", "")
        sql_query = state.get("sql_query", "")
        
        if sql_results:
            sql_results_str = "\n".join([
                ", ".join([f"{col}: {row.get(col, '')}" for col in sql_columns])
                for row in sql_results
            ])
        elif route in ["sql", "hybrid"]:
            if sql_query:
                # SQL was supposed to run but got no results
                sql_results_str = f"SQL query executed but returned no rows. Query: {sql_query[:200]}"
            else:
                # SQL should have been generated but wasn't
                sql_results_str = f"ERROR: SQL query was not generated for {route} question. Question requires database query but no SQL was produced."
        else:
            sql_results_str = "No SQL query needed (RAG-only question)"
        
        # Format retrieved docs
        retrieved_docs_str = "\n\n".join([
            f"[{chunk['chunk_id']}]: {chunk['content']}"
            for chunk in chunks
        ])
        
        result = synthesizer(
            question=question,
            format_hint=format_hint,
            sql_results=sql_results_str,
            retrieved_docs=retrieved_docs_str
        )
        final_answer = result.final_answer
        citations_str = result.citations
        
        # Parse citations - clean up the format
        citations = []
        if citations_str:
            # Split by comma and clean
            raw_citations = [c.strip() for c in citations_str.split(",") if c.strip()]
            for cit in raw_citations:
                # Clean up citation format - remove quotes, brackets, and extra whitespace
                cit = cit.strip('"').strip("'").strip('[').strip(']').strip()
                if cit and cit not in citations:
                    citations.append(cit)
        
        # Extract table names from SQL query
        if sql_query:
            table_names = sql_tool.get_table_names()
            for table in table_names:
                # Check if table name appears in SQL (case insensitive)
                if table.lower() in sql_query.lower() or f'"{table}"' in sql_query:
                    if table not in citations:
                        citations.append(table)
        
        # Add chunk IDs from retrieved chunks (only if they were actually used)
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id and chunk.get("score", 0) > 0.1:
                if chunk_id not in citations:
                    citations.append(chunk_id)
        
        state["final_answer"] = final_answer
        state["citations"] = citations
        state["trace"].append(f"Synthesized answer: {final_answer[:50]}...")
        return state
    
    def should_repair(state: AgentState) -> Literal["repair", "end"]:
        """Decide if repair is needed."""
        repair_count = state.get("repair_count", 0)
        
        if repair_count >= 2:
            return "end"
        
        # Check for SQL errors
        if state.get("sql_error"):
            return "repair"
        
        # Check if answer is empty
        if not state.get("final_answer"):
            return "repair"
        
        # Check format hint compliance (basic check)
        format_hint = state.get("format_hint", "")
        final_answer = state.get("final_answer", "")
        
        if "int" in format_hint.lower() and not final_answer.strip().isdigit():
            return "repair"
        
        return "end"
    
    def repair_node(state: AgentState) -> AgentState:
        """Repair by regenerating SQL or answer."""
        repair_count = state.get("repair_count", 0)
        state["repair_count"] = repair_count + 1

        # If SQL error, clear SQL to force regeneration
        sql_error = state.get("sql_error", "")
        if sql_error:
            state["trace"].append(f"Repair {repair_count + 1}: SQL error detected: {sql_error[:100]}")

            # For first repair attempt, try quick fixes
            # For second attempt, force regeneration
            if repair_count == 0:
                # Try to fix common SQL errors quickly
                sql_query = state.get("sql_query", "")
                if sql_query:
                    # Remove DSPy output artifacts
                    sql_query = re.sub(r':\d+\s*\n', '\n', sql_query)
                    sql_query = re.sub(r':\d+\s*$', '', sql_query)
                    sql_query = re.sub(r':\d+(\s+|,|\))', r'\1', sql_query)

                    # Fix common issues
                    sql_query = re.sub(r'\bBETWEENEN\b', 'BETWEEN', sql_query)
                    sql_query = re.sub(r'\bBETWE\b', 'BETWEEN', sql_query)
                    sql_query = re.sub(r'\bBETWEWS\b', 'BETWEEN', sql_query)  # Fix BETWEWS typo
                    sql_query = sql_query.replace('OrderDetails', '"Order Details"')
                    sql_query = sql_query.replace('orderdetails', '"Order Details"')
                    sql_query = sql_query.replace('Order_Details', '"Order Details"')
                    # Fix JOIN after WHERE errors
                    sql_query = re.sub(r'WHERE\s+(.*?)\s+JOIN', r'JOIN', sql_query, flags=re.IGNORECASE | re.DOTALL)
                    # Remove triple/double quotes
                    sql_query = re.sub(r"'''(\d{4}-\d{2}-\d{2})'''", r"'\1'", sql_query)
                    sql_query = re.sub(r"''(\d{4}-\d{2}-\d{2})''", r"'\1'", sql_query)
                    # Quote date strings (only if not already quoted)
                    sql_query = re.sub(r"(?<!')(\d{4}-\d{2}-\d{2})(?!')", r"'\1'", sql_query)

                    state["sql_query"] = sql_query
                    state["sql_error"] = ""  # Clear error to retry
                    state["trace"].append(f"Quick-fixed SQL: {sql_query[:100]}...")
                else:
                    # No SQL to fix, force regeneration
                    state["sql_query"] = ""
                    state["sql_error"] = ""
                    state["trace"].append("No SQL found, will regenerate")
            else:
                # Second repair: Force complete regeneration
                state["sql_query"] = ""
                state["sql_error"] = ""
                state["trace"].append("Forcing SQL regeneration after failed repair")

            return state

        # Otherwise, regenerate answer
        state["trace"].append(f"Repair {repair_count + 1}: Regenerating answer")
        state["final_answer"] = ""
        return state
    
    def validate_output(state: AgentState) -> AgentState:
        """Validate that output matches format_hint."""
        final_answer = state.get("final_answer", "")
        format_hint = state.get("format_hint", "")

        # Skip validation if empty
        if not final_answer:
            state["trace"].append("Validation: Empty answer")
            return state

        # Basic validation based on format_hint
        is_valid = True

        if format_hint == "int":
            # Check if it's a number
            if not str(final_answer).strip().replace('-', '').isdigit():
                is_valid = False
                state["trace"].append(f"Validation failed: Expected int, got {type(final_answer)}")

        elif format_hint == "float":
            # Check if it's a number
            try:
                float(final_answer)
            except:
                is_valid = False
                state["trace"].append(f"Validation failed: Expected float, got {final_answer}")

        elif format_hint.startswith("{") and format_hint.endswith("}"):
            # Should be a string that looks like JSON or a dict
            if not (isinstance(final_answer, dict) or (isinstance(final_answer, str) and "{" in final_answer)):
                is_valid = False
                state["trace"].append(f"Validation failed: Expected dict-like, got {type(final_answer)}")

        elif format_hint.startswith("list["):
            # Should be a list or string that looks like JSON array
            if not (isinstance(final_answer, list) or (isinstance(final_answer, str) and "[" in final_answer)):
                is_valid = False
                state["trace"].append(f"Validation failed: Expected list, got {type(final_answer)}")

        if is_valid:
            state["trace"].append("Validation: Output format OK")

        return state

    def calculate_confidence(state: AgentState) -> AgentState:
        """Calculate confidence score."""
        confidence = 0.5  # Base confidence

        # Boost if retrieval scores are high
        chunks = state.get("retrieved_chunks", [])
        if chunks:
            avg_score = sum(c["score"] for c in chunks) / len(chunks)
            confidence += avg_score * 0.3

        # Boost if SQL executed successfully
        if not state.get("sql_error") and state.get("sql_results"):
            confidence += 0.2

        # Reduce if repaired
        repair_count = state.get("repair_count", 0)
        confidence -= repair_count * 0.1

        state["confidence"] = max(0.0, min(1.0, confidence))
        return state
    
    # Build graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("nl_to_sql", nl_to_sql_node)
    workflow.add_node("executor", executor_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("repair", repair_node)
    workflow.add_node("validate_output", validate_output)
    workflow.add_node("calculate_confidence", calculate_confidence)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add edges based on route
    workflow.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {
            "rag": "retriever",
            "sql": "nl_to_sql",
            "hybrid": "retriever"
        }
    )
    
    # After retriever, go to planner if hybrid, else to synthesizer
    def route_after_retriever(state: AgentState) -> str:
        return "planner" if state["route"] == "hybrid" else "synthesizer"
    
    workflow.add_conditional_edges(
        "retriever",
        route_after_retriever,
        {
            "planner": "planner",
            "synthesizer": "synthesizer"
        }
    )
    
    # Planner -> NL-to-SQL
    workflow.add_edge("planner", "nl_to_sql")
    
    # NL-to-SQL -> Executor
    workflow.add_edge("nl_to_sql", "executor")
    
    # Executor -> Synthesizer
    workflow.add_edge("executor", "synthesizer")
    
    # Synthesizer -> Validate output
    workflow.add_edge("synthesizer", "validate_output")

    # Validate output -> Check if repair needed
    workflow.add_conditional_edges(
        "validate_output",
        should_repair,
        {
            "repair": "repair",
            "end": "calculate_confidence"
        }
    )
    
    # Repair -> Back to appropriate node
    def route_after_repair(state: AgentState) -> str:
        return "nl_to_sql" if state.get("sql_error") else "synthesizer"
    
    workflow.add_conditional_edges(
        "repair",
        route_after_repair,
        {
            "nl_to_sql": "nl_to_sql",
            "synthesizer": "synthesizer"
        }
    )
    
    # Confidence -> END
    workflow.add_edge("calculate_confidence", END)
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app

