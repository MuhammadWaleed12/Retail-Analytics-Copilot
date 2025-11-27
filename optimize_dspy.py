"""DSPy optimization script for NL-to-SQL module."""
import dspy
from agent.dspy_signatures import NLToSQL, NLToSQLSignature
from agent.tools.sqlite_tool import SQLiteTool


def create_training_set():
    """Create a small training set for NL-to-SQL optimization."""
    training_examples = [
        {
            "question": "What is the total revenue from Beverages category?",
            "schema": "Products(ProductID, ProductName, CategoryID), Order Details(OrderID, ProductID, UnitPrice, Quantity, Discount), Categories(CategoryID, CategoryName)",
            "constraints": '{"categories": ["Beverages"]}',
            "sql_query": 'SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as revenue FROM "Order Details" od JOIN Products p ON od.ProductID = p.ProductID JOIN Categories c ON p.CategoryID = c.CategoryID WHERE c.CategoryName = "Beverages"'
        },
        {
            "question": "Top 3 products by revenue",
            "schema": "Products(ProductID, ProductName), Order Details(OrderID, ProductID, UnitPrice, Quantity, Discount)",
            "constraints": "{}",
            "sql_query": 'SELECT p.ProductName, SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) as revenue FROM "Order Details" od JOIN Products p ON od.ProductID = p.ProductID GROUP BY p.ProductID, p.ProductName ORDER BY revenue DESC LIMIT 3'
        },
        {
            "question": "Average order value in December 1997",
            "schema": "Orders(OrderID, OrderDate), Order Details(OrderID, UnitPrice, Quantity, Discount)",
            "constraints": '{"date_ranges": [("1997-12-01", "1997-12-31")]}',
            "sql_query": 'SELECT SUM(od.UnitPrice * od.Quantity * (1 - od.Discount)) / COUNT(DISTINCT o.OrderID) as aov FROM Orders o JOIN "Order Details" od ON o.OrderID = od.OrderID WHERE o.OrderDate >= "1997-12-01" AND o.OrderDate <= "1997-12-31"'
        },
        # Add more examples...
    ]
    return training_examples


def evaluate_sql_module(module, test_cases):
    """Evaluate SQL module on test cases."""
    sql_tool = SQLiteTool("data/northwind.sqlite")
    schema = sql_tool.get_schema()
    
    valid_count = 0
    total = len(test_cases)
    
    for case in test_cases:
        sql_query = module.forward(
            question=case["question"],
            schema=schema,
            constraints=case.get("constraints", "")
        )
        
        # Clean SQL
        import re
        sql_query = re.sub(r'```sql\n?', '', sql_query)
        sql_query = re.sub(r'```\n?', '', sql_query)
        sql_query = sql_query.strip()
        
        # Try to execute
        _, _, error = sql_tool.execute_query(sql_query)
        if not error:
            valid_count += 1
    
    return valid_count / total if total > 0 else 0.0


def optimize_nl_to_sql():
    """Optimize NL-to-SQL module using BootstrapFewShot."""
    print("Initializing DSPy...")
    
    # Configure DSPy (assumes Ollama is set up)
    try:
        from langchain_ollama import OllamaLLM
        from dspy.langchain import LangChainLM
        
        ollama_llm = OllamaLLM(model="phi3.5:3.8b-mini-instruct-q4_K_M")
        lm = LangChainLM(ollama_llm)
        dspy.configure(lm=lm)
    except:
        lm = dspy.LM(model="phi3.5:3.8b-mini-instruct-q4_K_M")
        dspy.configure(lm=lm)
    
    # Create base module
    base_module = NLToSQL()
    
    # Evaluate before optimization
    print("Evaluating before optimization...")
    training_set = create_training_set()
    before_score = evaluate_sql_module(base_module, training_set[:3])  # Use subset for eval
    print(f"Before optimization: {before_score:.2%} valid SQL rate")
    
    # Optimize
    print("Optimizing with BootstrapFewShot...")
    from dspy.teleprompt import BootstrapFewShot
    
    optimizer = BootstrapFewShot(max_bootstrapped_demos=4, max_labeled_demos=8)
    optimized_module = optimizer.compile(
        student=base_module,
        trainset=training_set
    )
    
    # Evaluate after optimization
    print("Evaluating after optimization...")
    after_score = evaluate_sql_module(optimized_module, training_set[:3])
    print(f"After optimization: {after_score:.2%} valid SQL rate")
    
    print(f"\nImprovement: {after_score - before_score:.2%}")
    
    return optimized_module


if __name__ == "__main__":
    optimized = optimize_nl_to_sql()
    print("\nOptimization complete!")




