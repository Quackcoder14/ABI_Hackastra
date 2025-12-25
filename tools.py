import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import IsolationForest
import traceback
import streamlit as st

# --- 1. Data Loading and Helpers ---

def load_data():
    """
    Loads all dataframes from CSV files with the new relational structure.
    Returns a dictionary of dataframes for easy access.
    """
    try:
        # Load all four tables
        customers_df = pd.read_csv('customers.csv')
        orders_df = pd.read_csv('orders.csv')
        products_df = pd.read_csv('products.csv')
        revenue_df = pd.read_csv('revenue.csv')

        # Type conversions for better analysis
        orders_df['order_date'] = pd.to_datetime(orders_df['order_date'])
        orders_df['est_delivery'] = pd.to_datetime(orders_df['est_delivery'], errors='coerce')
        revenue_df['date'] = pd.to_datetime(revenue_df['date'])

        return {
            "customers_df": customers_df,
            "orders_df": orders_df,
            "products_df": products_df,
            "revenue_df": revenue_df
        }
    except FileNotFoundError as e:
        return f"Error: Required data file not found: {e}"
    except Exception as e:
        return f"Error loading data: {e}"

def get_data_schema():
    """Helper to get column names and relationships to feed into the System Prompt"""
    data_map = load_data()
    if isinstance(data_map, str): 
        return data_map
    
    schema_info = """
DATABASE SCHEMA (Relational Structure):

1. **customers_df**
   Columns: customer_id (PK), name, email, region
   Description: Customer master data
   
2. **products_df**
   Columns: product_id (PK), name, category, price, stock_level
   Description: Product catalog
   
3. **orders_df** (Bridge Table)
   Columns: order_id (PK), customer_id (FK), product_id (FK), status, order_date, est_delivery
   Description: Links customers to products; tracks order lifecycle
   Relationships:
   - customer_id → customers_df.customer_id
   - product_id → products_df.product_id
   
4. **revenue_df**
   Columns: revenue_id (PK), order_id (FK), amount, date, payment_method
   Description: Financial transactions linked to orders
   Relationships:
   - order_id → orders_df.order_id

IMPORTANT JOIN PATTERNS:
- Customer + Orders: orders_df.merge(customers_df, on='customer_id')
- Orders + Products: orders_df.merge(products_df, on='product_id')
- Orders + Revenue: orders_df.merge(revenue_df, on='order_id')
- Full Analysis: orders_df.merge(customers_df, on='customer_id').merge(products_df, on='product_id').merge(revenue_df, on='order_id')
"""
    return schema_info

def format_dataframe_output(df, max_rows=10):
    """
    Formats a pandas DataFrame into a clean, readable string format.
    
    Args:
        df: DataFrame to format
        max_rows: Maximum number of rows to display
    
    Returns:
        Formatted string representation
    """
    if isinstance(df, pd.Series):
        df = df.to_frame()
    
    if not isinstance(df, pd.DataFrame):
        return str(df)
    
    total_rows = len(df)
    display_df = df.head(max_rows)
    
    # Create header
    output = f"📊 Total Records: {total_rows}\n"
    if total_rows > max_rows:
        output += f"📋 Showing first {max_rows} rows\n"
    output += "─" * 70 + "\n\n"
    
    # Format each row
    for idx, row in display_df.iterrows():
        output += f"▸ Record {idx + 1 if isinstance(idx, int) else idx}:\n"
        for col in display_df.columns:
            value = row[col]
            # Format dates nicely
            if pd.api.types.is_datetime64_any_dtype(display_df[col]):
                value = pd.to_datetime(value).strftime('%Y-%m-%d') if pd.notna(value) else 'N/A'
            # Format floats to 2 decimal places
            elif isinstance(value, float):
                value = f"${value:,.2f}" if 'amount' in col.lower() or 'price' in col.lower() or 'revenue' in col.lower() else f"{value:.2f}"
            # Handle NaN
            elif pd.isna(value):
                value = 'N/A'
            
            output += f"  • {col}: {value}\n"
        output += "\n"
    
    if total_rows > max_rows:
        output += f"⋯ and {total_rows - max_rows} more records\n"
    
    return output

def format_series_output(series, name="Value"):
    """
    Formats a pandas Series into a clean, readable format.
    
    Args:
        series: Series to format
        name: Name for the value column
    
    Returns:
        Formatted string
    """
    output = f"📊 Analysis Results ({len(series)} items)\n"
    output += "─" * 70 + "\n\n"
    
    for idx, value in series.items():
        # Format the value
        if isinstance(value, float):
            formatted_value = f"${value:,.2f}" if 'amount' in name.lower() or 'revenue' in name.lower() else f"{value:,.2f}"
        else:
            formatted_value = str(value)
        
        output += f"▸ {idx}: {formatted_value}\n"
    
    return output

def format_scalar_output(value, description="Result"):
    """
    Formats a scalar value nicely.
    
    Args:
        value: The scalar value
        description: Description of the value
    
    Returns:
        Formatted string
    """
    output = f"📊 {description}\n"
    output += "─" * 70 + "\n\n"
    
    if isinstance(value, (int, float)):
        if isinstance(value, float):
            formatted = f"${value:,.2f}"
        else:
            formatted = f"{value:,}"
        output += f"▸ {formatted}\n"
    else:
        output += f"▸ {value}\n"
    
    return output

# --- 2. CUSTOMER TOOL (Privacy-Protected) ---

def get_customer_orders(customer_id: str) -> str:
    """
    Returns ALL order information for a specific customer, including product details.
    This tool filters data to show only the logged-in customer's orders.
    
    Args:
        customer_id: The customer's unique ID (e.g., CUST_001)
    
    Returns:
        Formatted string with customer's orders and product details
    """
    data = load_data()
    if isinstance(data, str): 
        return data
    
    customers_df = data['customers_df']
    orders_df = data['orders_df']
    products_df = data['products_df']
    
    # Clean the customer_id
    clean_id = str(customer_id).strip().upper()
    
    # Verify customer exists
    customer = customers_df[customers_df['customer_id'] == clean_id]
    if customer.empty:
        return f"ERROR: Customer ID '{clean_id}' not found in the system."
    
    # Get customer's orders with product details (JOIN)
    # Note: Using suffixes to avoid column name conflicts ('name' exists in both products and customers)
    customer_orders = orders_df[orders_df['customer_id'] == clean_id].merge(
        products_df, on='product_id', how='left', suffixes=('_order', '_product')
    )
    
    if customer_orders.empty:
        customer_name = customer['name'].iloc[0]
        return f"Hello {customer_name}! You currently have no orders in the system."
    
    # Format the response
    customer_name = customer['name'].iloc[0]
    customer_email = customer['email'].iloc[0]
    customer_region = customer['region'].iloc[0]
    
    result = f"👤 Customer: {customer_name} (ID: {clean_id})\n"
    result += f"📧 Email: {customer_email}\n"
    result += f"📍 Region: {customer_region}\n"
    result += f"📦 Total Orders: {len(customer_orders)}\n\n"
    result += "─" * 50 + "\n"
    
    for idx, row in customer_orders.iterrows():
        result += f"\n▸ Order ID: {row['order_id']}\n"
        result += f"  Product: {row['name']} ({row['category']})\n"
        result += f"  Price: ${row['price']:.2f}\n"
        result += f"  Status: {row['status']}\n"
        result += f"  Order Date: {row['order_date'].strftime('%Y-%m-%d')}\n"
        
        est_delivery = row['est_delivery']
        if pd.notna(est_delivery):
            result += f"  Est. Delivery: {est_delivery.strftime('%Y-%m-%d')}\n"
            
            # Check for delays
            if row['status'] != 'Delivered' and datetime.now() > est_delivery:
                days_late = (datetime.now() - est_delivery).days
                result += f"  ⚠️ DELAYED: {days_late} day(s) overdue\n"
        else:
            result += f"  Est. Delivery: N/A\n"
        
        result += "─" * 50 + "\n"
    
    return result


# --- 3. BUSINESS TOOL (Full Access) ---

def execute_pandas_code_business(python_code: str) -> str:
    """
    Executes Python code for business analytics with access to ALL tables.
    The code must calculate a result and store it in a variable named 'result'.
    
    Available dataframes:
    - customers_df: Customer master data (columns: customer_id, name, email, region)
    - orders_df: Order transactions (columns: order_id, customer_id, product_id, status, order_date, est_delivery)
    - products_df: Product catalog (columns: product_id, name, category, price, stock_level)
    - revenue_df: Financial transactions (columns: revenue_id, order_id, amount, date, payment_method)
    
    IMPORTANT: When joining tables that both have a 'name' column (customers and products),
    use suffixes parameter to avoid conflicts:
    orders_df.merge(products_df, on='product_id', suffixes=('_order', '_product'))
    
    Example for cross-table analysis:
    ```python
    # Revenue by customer region
    order_revenue = orders_df.merge(revenue_df, on='order_id')
    regional_data = order_revenue.merge(customers_df, on='customer_id')
    result = regional_data.groupby('region')['amount'].sum().sort_values(ascending=False).to_string()
    ```
    
    Args:
        python_code: Python code string to execute
    
    Returns:
        String result of the computation
    """
    data_map = load_data()
    if isinstance(data_map, str): 
        return data_map

    # Create execution environment with all dataframes
    local_env = data_map.copy()
    local_env['pd'] = pd
    local_env['datetime'] = datetime
    local_env['format_dataframe_output'] = format_dataframe_output
    local_env['format_series_output'] = format_series_output
    local_env['format_scalar_output'] = format_scalar_output

    try:
        # Execute the code
        exec(python_code, {}, local_env)
        
        if 'result' in local_env:
            result_value = local_env['result']
            
            # Handle None explicitly
            if result_value is None:
                return "⚠️ Warning: The calculation returned None. Please check your code logic."
            
            # Format the output based on type
            if isinstance(result_value, pd.DataFrame):
                return format_dataframe_output(result_value)
            elif isinstance(result_value, pd.Series):
                return format_series_output(result_value)
            elif isinstance(result_value, (int, float, np.integer, np.floating)):
                return format_scalar_output(result_value)
            else:
                # For other types, just convert to string
                return str(result_value)
        else:
            return "❌ Error: The code ran, but no variable named 'result' was defined. Please assign the final answer to 'result'."
            
    except Exception as e:
        error_msg = f"Python Execution Error: {str(e)}\n"
        error_msg += f"Traceback: {traceback.format_exc()}\n\n"
        error_msg += "Common issues:\n"
        error_msg += "- Check column names match the schema exactly (case-sensitive)\n"
        error_msg += "- Ensure you're using correct merge keys (customer_id, product_id, order_id)\n"
        error_msg += "- Use suffixes when joining tables with duplicate column names (e.g., 'name' in customers and products)\n"
        error_msg += "- Verify dataframe names: customers_df, orders_df, products_df, revenue_df"
        return error_msg


# --- 4. Proactive Audit Tools (Business Only) ---

def check_for_revenue_anomalies() -> str:
    """
    Runs Isolation Forest on revenue data to detect unusual patterns.
    Business dashboard feature for proactive monitoring.
    """
    data = load_data()
    if isinstance(data, str): 
        return data
    revenue_df = data['revenue_df']
    
    if len(revenue_df) < 5:
        return "Not enough data points to run anomaly detection."

    X = revenue_df[['amount']].values
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X)
    revenue_df['anomaly'] = model.predict(X)
    anomalies = revenue_df[revenue_df['anomaly'] == -1]
    
    if not anomalies.empty:
        latest_anomaly = anomalies.iloc[-1]
        return f"CRITICAL REVENUE ANOMALY: Unusual pattern on {latest_anomaly['date'].strftime('%Y-%m-%d')} - Amount: ${latest_anomaly['amount']:,.2f} (Order: {latest_anomaly['order_id']})."
    else:
        return "SUCCESS: No significant revenue anomalies detected."


def check_for_critical_delays() -> str:
    """
    Checks for past-due orders across all customers.
    Business dashboard feature for logistics monitoring.
    """
    data = load_data()
    if isinstance(data, str): 
        return data
    orders_df = data['orders_df']
    customers_df = data['customers_df']
    
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    
    # Find delayed orders
    critically_delayed_orders = orders_df[
        (orders_df['est_delivery'] < today) & 
        (orders_df['status'] != 'Delivered') &
        (orders_df['status'] != 'Cancelled') &
        (pd.notna(orders_df['est_delivery']))
    ]
    
    if not critically_delayed_orders.empty:
        count = len(critically_delayed_orders)
        
        # Join with customers to show affected customers
        delayed_with_customers = critically_delayed_orders.merge(
            customers_df, on='customer_id', how='left'
        )
        
        # Convert to strings explicitly
        order_list = [str(oid) for oid in critically_delayed_orders['order_id'].tolist()[:3]]
        customer_names = [str(name) for name in delayed_with_customers['name'].tolist()[:3]]
        
        return f"ALERT: {count} orders are critically delayed!\nAffected Orders: {', '.join(order_list)}\nAffected Customers: {', '.join(customer_names)}"
    else:
        return "SUCCESS: No critical delivery delays found."


# --- 5. Helper for Streamlit Integration ---

def get_customer_id_from_session():
    """
    Retrieves the customer_id from Streamlit session state.
    Used by the customer tool to enforce privacy.
    """
    if 'customer_id' in st.session_state:
        return st.session_state.customer_id
    return None

def check_customer_order_status(customer_id: str) -> dict:
    """
    Checks if a customer has any delayed orders.
    Returns a dict with status info for notifications.
    """
    data = load_data()
    if isinstance(data, str):
        return {"status": "error", "message": "Unable to load data"}
    
    orders_df = data['orders_df']
    
    clean_id = str(customer_id).strip().upper()
    customer_orders = orders_df[orders_df['customer_id'] == clean_id]
    
    if customer_orders.empty:
        return {"status": "normal", "message": "You have no active orders"}
    
    today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    
    # Check for delays
    delayed = customer_orders[
        (customer_orders['est_delivery'] < today) & 
        (customer_orders['status'] != 'Delivered') &
        (customer_orders['status'] != 'Cancelled') &
        (pd.notna(customer_orders['est_delivery']))
    ]
    
    if not delayed.empty:
        count = len(delayed)
        return {
            "status": "delayed",
            "message": f"⚠️ You have {count} delayed order{'s' if count > 1 else ''}!",
            "count": count
        }
    else:
        on_time_count = len(customer_orders[customer_orders['status'].isin(['Pending', 'Shipped'])])
        delivered_count = len(customer_orders[customer_orders['status'] == 'Delivered'])
        
        if on_time_count > 0:
            return {
                "status": "normal",
                "message": f"✅ All {on_time_count} active order{'s are' if on_time_count > 1 else ' is'} on track!",
                "count": on_time_count
            }
        else:
            return {
                "status": "normal",
                "message": f"✅ You have {delivered_count} completed order{'s' if delivered_count > 1 else ''}",
                "count": delivered_count
            }