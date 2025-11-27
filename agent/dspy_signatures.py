"""DSPy signatures for Router, NL-to-SQL, and Synthesizer."""
import dspy
from typing import Literal


class RouterSignature(dspy.Signature):
    """Classify question type: rag, sql, or hybrid.
    
    - rag: Questions about policies, definitions, or documents only (e.g., "what is the return policy?")
    - sql: Questions requiring database queries with numbers/calculations (e.g., "top 3 products by revenue", "total revenue")
    - hybrid: Questions needing both documents (for dates/formulas) AND database queries (e.g., "revenue during Summer Beverages campaign")
    """
    question: str = dspy.InputField(desc="User question")
    route: Literal["rag", "sql", "hybrid"] = dspy.OutputField(desc="Must be exactly: 'rag' OR 'sql' OR 'hybrid'. Choose 'sql' if question asks for numbers/calculations from database. Choose 'hybrid' if question needs both document info (dates/formulas) AND database query. Choose 'rag' only if question is purely about documents/policies with no numbers.")


class NLToSQLSignature(dspy.Signature):
    """Generate SQL query from natural language question.

    CRITICAL REQUIREMENTS:
    1. Generate ONLY valid SQLite syntax
    2. Table names with spaces must use double quotes: "Order Details"
    3. Date strings must use single quotes: '1997-06-01'
    4. Complete the entire query - no truncation
    5. Use proper JOIN syntax: FROM table1 JOIN table2 ON condition WHERE ...
    6. Revenue formula: SUM(UnitPrice * Quantity * (1 - Discount))
    7. Always end with semicolon

    Example:
    Question: "Top 3 products by revenue"
    SQL: SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) AS revenue FROM "Order Details" od JOIN Products p ON od.ProductID = p.ProductID GROUP BY p.ProductName ORDER BY revenue DESC LIMIT 3;
    """
    question: str = dspy.InputField(desc="User question")
    db_schema: str = dspy.InputField(desc="Database schema information (key tables only)")
    constraints: str = dspy.InputField(desc="Extracted constraints (dates, categories, KPIs)")
    sql_query: str = dspy.OutputField(desc="Valid SQLite SELECT query. Must be complete, syntactically correct, and use proper table/column names from schema. No markdown, no truncation.")


class SynthesizerSignature(dspy.Signature):
    """Synthesize final answer from SQL results and retrieved docs.

    FORMAT REQUIREMENTS:
    - int: Return just the number (e.g., "14")
    - float: Return just the number with 2 decimals (e.g., "123.45")
    - {key:type, key:type}: Return valid JSON object (e.g., '{"category": "Beverages", "quantity": 150}')
    - list[{key:type}]: Return valid JSON array (e.g., '[{"product": "Chai", "revenue": 123.45}]')

    IMPORTANT:
    - Use ONLY data from sql_results or retrieved_docs - never invent numbers
    - For dict/list formats, output valid JSON that can be parsed with json.loads()
    - Do not include any explanatory text, just the answer value

    Examples:
    format_hint="int", question about return days → "14"
    format_hint="float", AOV question → "543.21"
    format_hint="{category:str, quantity:int}" → '{"category": "Beverages", "quantity": 150}'
    """
    question: str = dspy.InputField(desc="Original user question")
    format_hint: str = dspy.InputField(desc="Expected output format (e.g., int, float, {category:str, quantity:int})")
    sql_results: str = dspy.InputField(desc="SQL query results. Use these exact numbers if present.")
    retrieved_docs: str = dspy.InputField(desc="Retrieved document chunks")
    final_answer: str = dspy.OutputField(desc="Answer value ONLY in exact format specified by format_hint. For JSON formats, return parseable JSON string.")
    citations: str = dspy.OutputField(desc="Comma-separated list WITHOUT brackets: Orders, Products, kpi_definitions::chunk0 (no surrounding quotes or brackets)")


class Router(dspy.Module):
    """Router module to classify question type."""
    
    def __init__(self):
        super().__init__()
        self.classify = dspy.ChainOfThought(RouterSignature)
    
    def forward(self, question: str):
        """Route question to appropriate handler."""
        result = self.classify(question=question)
        # Return the result object so we can access .route
        return result


class NLToSQL(dspy.Module):
    """Natural language to SQL converter."""
    
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(NLToSQLSignature)
    
    def forward(self, question: str, schema: str, constraints: str = ""):
        """Generate SQL query from natural language."""
        result = self.generate(
            question=question,
            db_schema=schema,
            constraints=constraints
        )
        # Return the result object so we can access .sql_query
        return result


class Synthesizer(dspy.Module):
    """Synthesize final answer from multiple sources."""
    
    def __init__(self):
        super().__init__()
        self.synthesize = dspy.ChainOfThought(SynthesizerSignature)
    
    def forward(self, question: str, format_hint: str, sql_results: str, retrieved_docs: str):
        """Synthesize answer and citations."""
        result = self.synthesize(
            question=question,
            format_hint=format_hint,
            sql_results=sql_results,
            retrieved_docs=retrieved_docs
        )
        # Return the result object so we can access .final_answer and .citations
        return result




