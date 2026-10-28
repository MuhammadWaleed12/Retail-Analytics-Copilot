"""SQLite tool for database access and schema introspection."""
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class SQLiteTool:
    """Tool for executing read-only SQL queries and introspecting SQLite schema."""

    def __init__(
        self,
        db_path: str,
        max_rows: int = 1_000,
        max_vm_steps: int = 1_000_000,
        progress_interval: int = 1_000,
    ):
        """Initialize with path to SQLite database."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        if max_rows < 1:
            raise ValueError("max_rows must be at least 1")
        if max_vm_steps < 1:
            raise ValueError("max_vm_steps must be at least 1")
        if progress_interval < 1:
            raise ValueError("progress_interval must be at least 1")
        self.max_rows = max_rows
        self.max_vm_steps = max_vm_steps
        self.progress_interval = progress_interval

    def _connect(self) -> sqlite3.Connection:
        """Open a connection that cannot mutate the source database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA query_only = ON")
        return conn
    
    def get_schema(self) -> str:
        """Get database schema using PRAGMA statements."""
        conn = self._connect()
        cursor = conn.cursor()
        
        schema_parts = []
        
        # Get only relevant table names (filter out system tables)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        
        # Focus on main tables used in queries
        main_tables = ["Orders", "Order Details", "Products", "Customers", "Categories", "Suppliers"]
        tables = [t for t in tables if t in main_tables] or tables[:6]  # Limit to 6 most relevant
        
        for table in tables:
            # Quote table name if it contains spaces or special characters
            quoted_table = f'"{table}"' if ' ' in table or '-' in table else table
            
            # Get table schema
            cursor.execute(f"PRAGMA table_info({quoted_table})")
            columns = cursor.fetchall()
            
            schema_parts.append(f"\nTable: {table}")
            schema_parts.append("Columns:")
            for col in columns[:10]:  # Limit columns to first 10
                col_name, col_type = col[1], col[2]
                schema_parts.append(f"  - {col_name} ({col_type})")
        
        conn.close()
        return "\n".join(schema_parts)
    
    def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], List[str], Optional[str]]:
        """
        Execute a read-only SQL query and return results.
        
        Returns:
            Tuple of (rows as dicts, column names, error message if any)
        """
        conn = self._connect()
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        cursor = conn.cursor()
        executed_steps = 0
        budget_exceeded = False

        def enforce_instruction_budget() -> int:
            """Abort model-generated queries that exceed the configured budget."""
            nonlocal executed_steps, budget_exceeded
            executed_steps += self.progress_interval
            if executed_steps >= self.max_vm_steps:
                budget_exceeded = True
                return 1
            return 0

        conn.set_progress_handler(enforce_instruction_budget, self.progress_interval)

        try:
            cursor.execute(query)

            # SQLite metadata identifies result sets, including WITH queries,
            # commented SELECTs, and writes with a RETURNING clause.
            if cursor.description is not None:
                rows = cursor.fetchmany(self.max_rows + 1)
                columns = [description[0] for description in cursor.description]
                if len(rows) > self.max_rows:
                    return (
                        [],
                        columns,
                        f"Query returned more than {self.max_rows} rows; add filters or LIMIT",
                    )
                result = [dict(row) for row in rows]
                return result, columns, None
            return [], [], None
        except Exception as e:
            if budget_exceeded:
                error_msg = (
                    "Query execution interrupted after exceeding the "
                    f"{self.max_vm_steps}-instruction budget"
                )
            else:
                error_msg = str(e)
            return [], [], error_msg
        finally:
            conn.set_progress_handler(None, 0)
            conn.close()
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names."""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables



