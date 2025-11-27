"""Main CLI entrypoint for the hybrid agent."""
import json
import ast
import re
from pathlib import Path
import click
from rich.console import Console
from rich.progress import Progress

from agent.graph_hybrid import create_hybrid_agent


console = Console()


def parse_answer(answer: str, format_hint: str):
    """Parse answer string to match format_hint exactly."""
    answer = answer.strip()

    # Remove markdown code blocks if present
    answer = re.sub(r'```[a-z]*\n?', '', answer)
    answer = answer.strip()

    # Remove any explanatory text before/after the actual answer
    # Try to extract just the value part

    if format_hint == "int":
        # Extract integer - look for first number
        match = re.search(r'\b(\d+)\b', answer)
        if match:
            return int(match.group(1))
        return 0

    elif format_hint == "float":
        # Extract float - look for decimal number
        match = re.search(r'\b(\d+\.?\d*)\b', answer)
        if match:
            return round(float(match.group(1)), 2)
        return 0.0

    elif format_hint.startswith("{") and format_hint.endswith("}"):
        # Parse dict-like structure
        # First try: direct JSON parsing
        try:
            # Look for JSON object in the answer
            json_match = re.search(r'\{[^}]+\}', answer)
            if json_match:
                json_str = json_match.group()
                # Try json.loads first (handles proper JSON)
                try:
                    parsed = json.loads(json_str)
                    # Ensure float values are rounded to 2 decimals
                    for key, val in parsed.items():
                        if isinstance(val, float):
                            parsed[key] = round(val, 2)
                    return parsed
                except:
                    # Fall back to ast.literal_eval
                    parsed = ast.literal_eval(json_str)
                    for key, val in parsed.items():
                        if isinstance(val, float):
                            parsed[key] = round(val, 2)
                    return parsed
        except Exception as e:
            pass

        # Second try: extract key-value pairs manually
        result = {}
        if "category" in format_hint.lower():
            cat_match = re.search(r'category["\']?\s*[:=]\s*["\']?([^,"\'}\]]+)', answer, re.I)
            if cat_match:
                result["category"] = cat_match.group(1).strip()

        if "quantity" in format_hint.lower():
            qty_match = re.search(r'quantity["\']?\s*[:=]\s*(\d+)', answer, re.I)
            if qty_match:
                result["quantity"] = int(qty_match.group(1))

        if "customer" in format_hint.lower():
            cust_match = re.search(r'customer["\']?\s*[:=]\s*["\']?([^,"\'}\]]+)', answer, re.I)
            if cust_match:
                result["customer"] = cust_match.group(1).strip()

        if "margin" in format_hint.lower():
            margin_match = re.search(r'margin["\']?\s*[:=]\s*(\d+\.?\d*)', answer, re.I)
            if margin_match:
                result["margin"] = round(float(margin_match.group(1)), 2)

        if "revenue" in format_hint.lower():
            rev_match = re.search(r'revenue["\']?\s*[:=]\s*(\d+\.?\d*)', answer, re.I)
            if rev_match:
                result["revenue"] = round(float(rev_match.group(1)), 2)

        if "product" in format_hint.lower():
            prod_match = re.search(r'product["\']?\s*[:=]\s*["\']?([^,"\'}\]]+)', answer, re.I)
            if prod_match:
                result["product"] = prod_match.group(1).strip()

        return result if result else answer

    elif format_hint.startswith("list[") and "{" in format_hint:
        # Parse list of objects
        try:
            # Try to find JSON array
            json_match = re.search(r'\[.*\]', answer, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                # Try json.loads first
                try:
                    parsed = json.loads(json_str)
                    # Ensure float values are rounded
                    for item in parsed:
                        if isinstance(item, dict):
                            for key, val in item.items():
                                if isinstance(val, float):
                                    item[key] = round(val, 2)
                    return parsed
                except:
                    # Fall back to ast.literal_eval
                    parsed = ast.literal_eval(json_str)
                    for item in parsed:
                        if isinstance(item, dict):
                            for key, val in item.items():
                                if isinstance(val, float):
                                    item[key] = round(val, 2)
                    return parsed
        except Exception as e:
            # Last resort: return empty list
            return []

    return answer


@click.command()
@click.option("--batch", required=True, help="Path to JSONL file with questions")
@click.option("--out", required=True, help="Path to output JSONL file")
@click.option("--db", default="data/northwind.sqlite", help="Path to SQLite database")
@click.option("--docs", default="docs", help="Path to docs directory")
@click.option("--model", default="phi3.5:3.8b-mini-instruct-q4_K_M", help="Ollama model name")
def main(batch: str, out: str, db: str, docs: str, model: str):
    """Run the hybrid agent on a batch of questions."""
    
    # Validate paths
    batch_path = Path(batch)
    if not batch_path.exists():
        console.print(f"[red]Error: Batch file not found: {batch}[/red]")
        return
    
    db_path = Path(db)
    if not db_path.exists():
        console.print(f"[red]Error: Database not found: {db}[/red]")
        return
    
    docs_path = Path(docs)
    if not docs_path.exists():
        console.print(f"[red]Error: Docs directory not found: {docs}[/red]")
        return
    
    # Create agent
    console.print(f"[green]Initializing agent with model: {model}[/green]")
    agent = create_hybrid_agent(str(db_path), str(docs_path), model)
    
    # Load questions
    questions = []
    with open(batch_path, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    
    console.print(f"[green]Loaded {len(questions)} questions[/green]")
    
    # Process questions
    results = []
    
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing questions...", total=len(questions))
        
        for q in questions:
            q_id = q["id"]
            question = q["question"]
            format_hint = q["format_hint"]
            
            console.print(f"\n[cyan]Processing: {q_id}[/cyan]")
            console.print(f"Question: {question}")
            
            # Initialize state
            initial_state = {
                "question": question,
                "format_hint": format_hint,
                "route": "",
                "retrieved_chunks": [],
                "constraints": {},
                "sql_query": "",
                "sql_results": [],
                "sql_columns": [],
                "sql_error": "",
                "final_answer": "",
                "citations": [],
                "explanation": "",
                "confidence": 0.0,
                "repair_count": 0,
                "trace": []
            }
            
            # Run agent
            config = {"configurable": {"thread_id": q_id}}
            try:
                # Accumulate state updates
                current_state = initial_state.copy()
                node_count = 0
                for update in agent.stream(initial_state, config):
                    node_count += 1
                    node_name = list(update.keys())[0] if update else 'processing'
                    console.print(f"  [dim]Node {node_count}: {node_name}[/dim]")
                    
                    # Update state with latest changes
                    for node_name, node_state in update.items():
                        if isinstance(node_state, dict):
                            current_state.update(node_state)
                            # Debug: show route and SQL query status
                            if node_name == "router":
                                console.print(f"    [blue]Route: {node_state.get('route', 'unknown')}[/blue]")
                            elif node_name == "nl_to_sql":
                                sql = node_state.get('sql_query', '')
                                if sql:
                                    console.print(f"    [green]SQL generated: {sql[:80]}...[/green]")
                                else:
                                    console.print(f"    [yellow]Warning: No SQL generated[/yellow]")
                            elif node_name == "executor":
                                error = node_state.get('sql_error', '')
                                rows = len(node_state.get('sql_results', []))
                                if error:
                                    console.print(f"    [red]SQL error: {error[:80]}[/red]")
                                else:
                                    console.print(f"    [green]SQL executed: {rows} rows[/green]")
                    
                    # Safety: prevent infinite loops
                    if node_count > 50:
                        console.print(f"  [yellow]Warning: Too many nodes ({node_count}), stopping[/yellow]")
                        break
                
                # Get final state
                final_answer_raw = current_state.get("final_answer", "")
                citations = current_state.get("citations", [])
                sql_query = current_state.get("sql_query", "")
                confidence = current_state.get("confidence", 0.0)
                trace = current_state.get("trace", [])
                
                # Parse answer
                final_answer = parse_answer(final_answer_raw, format_hint)
                
                # Generate explanation
                explanation = f"Processed via {current_state.get('route', 'unknown')} route"
                if trace:
                    explanation = trace[-1] if trace else explanation
                
                # Ensure SQL is captured even if empty
                sql_query_final = current_state.get("sql_query", "")
                
                result = {
                    "id": q_id,
                    "final_answer": final_answer,
                    "sql": sql_query_final if sql_query_final else "",
                    "confidence": round(confidence, 2),
                    "explanation": explanation[:200] if explanation else "Processed",
                    "citations": citations if citations else []
                }
                
            except Exception as e:
                console.print(f"[red]Error processing {q_id}: {e}[/red]")
                result = {
                    "id": q_id,
                    "final_answer": "",
                    "sql": "",
                    "confidence": 0.0,
                    "explanation": f"Error: {str(e)}",
                    "citations": []
                }
            
            results.append(result)
            progress.update(task, advance=1)
    
    # Write results
    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    console.print(f"\n[green]Results written to: {output_path}[/green]")
    console.print(f"[green]Processed {len(results)} questions[/green]")


if __name__ == "__main__":
    main()

