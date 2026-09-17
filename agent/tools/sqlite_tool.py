"""SQLite tool for database access and schema introspection."""
import sqlite3
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class SQLiteTool:
    """Tool for executing SQL queries and introspecting SQLite schema."""
    
    def __init__(self, db_path: str):
        """Initialize with path to SQLite database."""
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def get_schema(self) -> str:
        """Get database schema using PRAGMA statements."""
        conn = sqlite3.connect(self.db_path)
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
        Execute SQL query and return results.
        
        Returns:
            Tuple of (rows as dicts, column names, error message if any)
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        cursor = conn.cursor()
        
        try:
            cursor.execute(query)
            
            # SQLite metadata identifies result sets, including WITH queries,
            # commented SELECTs, and writes with a RETURNING clause.
            if cursor.description is not None:
                rows = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                result = [dict(row) for row in rows]
                # Consume RETURNING rows before committing the write.
                conn.commit()
                conn.close()
                return result, columns, None
            else:
                conn.commit()
                conn.close()
                return [], [], None
        except Exception as e:
            error_msg = str(e)
            conn.close()
            return [], [], error_msg
    
    def get_table_names(self) -> List[str]:
        """Get list of all table names."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables



