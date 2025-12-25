import streamlit as st
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
from tools import (
    get_customer_orders, 
    execute_pandas_code_business,
    check_for_revenue_anomalies, 
    check_for_critical_delays,
    get_data_schema,
    check_customer_order_status
)

# --- Configuration and Initialization ---

load_dotenv()
try:
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    st.error(f"Error initializing Gemini client: {e}")
    st.stop()

# --- Load Credentials ---
CREDENTIALS_FILE = "credentials.json"

def load_credentials():
    try:
        if not os.path.exists(CREDENTIALS_FILE):
            with open(CREDENTIALS_FILE, "w") as f:
                json.dump({}, f)
        with open(CREDENTIALS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error("Error: 'credentials.json' is improperly formatted. Resetting to empty.")
        return {}
    except Exception as e:
        st.error(f"Error loading credentials: {e}")
        return {}

def save_credentials(data):
    try:
        with open(CREDENTIALS_FILE, "w") as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Error saving credentials: {e}")
        return False

CREDENTIALS = load_credentials()

# --- Tool Definitions ---
REGISTERED_CUSTOMER_TOOLS = [get_customer_orders]
REGISTERED_BUSINESS_TOOLS_ALL = [execute_pandas_code_business]

# --- System Instructions ---
DATA_SCHEMA = get_data_schema()

def get_customer_system_instruction():
    """Generate customer system instruction with current customer_id"""
    customer_id = st.session_state.get('customer_id', 'UNKNOWN')
    
    return f"""
You are the Customer Service Agent. Your persona is polite, concise, and focused on helping customers track their orders.

**IMPORTANT: The logged-in customer's ID is: {customer_id}**

You have access to the 'get_customer_orders' tool which shows all orders for the logged-in customer.

**CRITICAL INSTRUCTION FOR TOOL USAGE:**
- When the customer asks about their orders, ALWAYS call the tool with the exact customer_id: "{customer_id}"
- DO NOT ask the customer for their customer_id - you already know it
- DO NOT modify or concatenate the customer_id in any way
- Simply use: get_customer_orders(customer_id="{customer_id}")

The tool returns order details including:
- Order ID, Status, Order Date, Estimated Delivery
- Product Name, Category, Price
- Customer's own information

**INSTRUCTIONS:**
1. When a customer asks about their orders, immediately use 'get_customer_orders' with customer_id="{customer_id}"
2. Present the information in a friendly, easy-to-understand format.
3. If asked about specific order status, delays, or product details, extract from the tool's response.
4. NEVER discuss revenue, sales analytics, other customers' data, or business metrics.
5. If asked about business data, politely say: "I can only help with your order inquiries. For business analytics, please contact the business portal."

**Privacy Note:** You can only see the logged-in customer's own orders. This protects customer privacy.
"""

SYSTEM_INSTRUCTION_BUSINESS = f"""
You are the Autonomous Business Intelligence (ABI) Analyst. Your persona is professional, strategic, and highly analytical.

You have access to a powerful tool called 'execute_pandas_code_business' which allows you to run Python analysis on ALL company data across multiple tables with foreign key relationships.

{DATA_SCHEMA}

**DATABASE RELATIONSHIPS:**
- customers.customer_id ← orders.customer_id (one-to-many)
- products.product_id ← orders.product_id (one-to-many)
- orders.order_id ← revenue.order_id (one-to-one)

**CRITICAL: HANDLING DUPLICATE COLUMN NAMES**
Both 'customers_df' and 'products_df' have a column called 'name'. When joining these tables with orders_df, you MUST use the suffixes parameter to avoid conflicts:

```python
# CORRECT way to join when 'name' column exists in multiple tables:
orders_with_products = orders_df.merge(products_df, on='product_id', suffixes=('_order', '_product'))
orders_with_customers = orders_df.merge(customers_df, on='customer_id', suffixes=('_order', '_customer'))

# For full data with all tables:
full_data = orders_df.merge(
    customers_df, on='customer_id'
).merge(
    products_df, on='product_id', suffixes=('_customer', '_product')
).merge(
    revenue_df, on='order_id'
)
# Note: After first merge, 'name' from customers is preserved, then suffixes handle the products 'name'
```

**INSTRUCTIONS FOR USING THE TOOL:**
1. When asked ANY question about data, write Python Pandas code to answer it.
2. The dataframes are pre-loaded as: `customers_df`, `orders_df`, `products_df`, `revenue_df`.
3. Use pandas `.merge()` to combine tables using foreign keys.
4. ALWAYS use suffixes parameter when joining tables that share column names.
5. You MUST assign the final answer to a variable named `result`.
6. DO NOT use .to_string() method - just assign the DataFrame or Series directly to result.
7. Return clear, business-focused insights to the user.

**COMMON QUERY PATTERNS:**

Total revenue:
```python
result = revenue_df['amount'].sum()
```

Customer revenue analysis (with suffix handling):
```python
customer_orders = orders_df.merge(customers_df, on='customer_id')
customer_revenue = customer_orders.merge(revenue_df, on='order_id')
result = customer_revenue.groupby('name')['amount'].sum().sort_values(ascending=False).head(5)
```

Product performance by region (handling name conflicts):
```python
full_data = orders_df.merge(
    customers_df, on='customer_id'
).merge(
    products_df, on='product_id', suffixes=('_customer', '_product')
).merge(
    revenue_df, on='order_id'
)
result = full_data.groupby(['region', 'category'])['amount'].sum()
```

Top spending customers:
```python
customer_revenue = orders_df.merge(revenue_df, on='order_id').merge(customers_df, on='customer_id')
result = customer_revenue.groupby('name')['amount'].sum().sort_values(ascending=False).head(10)
```

Count of orders by status:
```python
result = orders_df['status'].value_counts()
```

**Remember:** 
- 'name' exists in BOTH customers_df and products_df
- Use suffixes to distinguish between them after joining
- Always check which 'name' column you need for your analysis
"""

st.set_page_config(layout="wide", page_title="ABI Agent: Intelligent Portal", page_icon="🤖")

# --- Session State Initialization ---
if "page" not in st.session_state:
    st.session_state.page = "selector" 
if "customer_history" not in st.session_state:
    st.session_state.customer_history = [] 
if "business_history" not in st.session_state:
    st.session_state.business_history = [] 
if "audit_log" not in st.session_state:
    st.session_state.audit_log = [] 
if "last_raw_response" not in st.session_state:
    st.session_state.last_raw_response = None
if 'revenue_alert_status' not in st.session_state:
    st.session_state.revenue_alert_status = "Pending (Run Audit)"
if 'delay_alert_status' not in st.session_state:
    st.session_state.delay_alert_status = "Pending (Run Audit)"
if 'auth_role_pending' not in st.session_state:
    st.session_state.auth_role_pending = None
if 'authenticated_user' not in st.session_state:
    st.session_state.authenticated_user = None
if 'customer_id' not in st.session_state:
    st.session_state.customer_id = None
if 'notification_shown' not in st.session_state:
    st.session_state.notification_shown = False

# --- STAR FIELD & LAYOUT CSS ---

STAR_FIELD_CSS = """
/* The space theme */
.stApp {
    background-color: #000000;
    background-attachment: fixed;
    background-image: 
        radial-gradient(circle at 80% 30%, #ffffff 0.5px, transparent 1.5px), 
        radial-gradient(circle at 15% 65%, #ffffff 0.7px, transparent 1.7px),
        radial-gradient(circle at 20% 70%, #fffacd 0.4px, transparent 1.2px), 
        radial-gradient(circle at 90% 40%, #fffacd 0.6px, transparent 1.6px),
        radial-gradient(circle at 50% 10%, rgba(255, 255, 255, 0.7) 0.5px, transparent 1px),
        radial-gradient(circle at 30% 90%, rgba(255, 255, 255, 0.5) 0.3px, transparent 0.8px),
        radial-gradient(circle at 70% 80%, rgba(255, 255, 255, 0.4) 0.2px, transparent 0.6px),
        radial-gradient(circle at 40% 20%, rgba(255, 255, 255, 0.3) 0.2px, transparent 0.5px);
    background-size: 600px 600px, 750px 750px, 900px 900px, 1100px 1100px, 1300px 1300px, 1500px 1500px, 1700px 1700px, 1900px 1900px;
    background-position: 0 0, 100px 100px, 200px 200px, 300px 300px, 400px 400px, 500px 500px, 600px 600px, 700px 700px;
    animation: move-stars 300s linear infinite;
}
@keyframes move-stars {
    from { background-position: 0 0, 100px 100px, 200px 200px, 300px 300px, 400px 400px, 500px 500px, 600px 600px, 700px 700px; }
    to { background-position: 600px 600px, 850px 850px, 1100px 1100px, 1400px 1400px, 1700px 1700px, 2000px 2000px, 2300px 2300px, 2600px 2600px; }
}
"""

st.markdown(f"""
<style>
    {STAR_FIELD_CSS}
    .main .block-container {{ padding-top: 0rem !important; padding-bottom: 2rem !important; max-width: 100%; }}
    .main .block-container > div:first-child:empty, .main .block-container > div[data-testid]:first-child {{ display: none !important; }}
    h1, h2, h3, h4, .stMarkdown, .stText, .stCaption {{ color: #FFFFFF !important; }}
    .stApp > div {{ color: #FFFFFF !important; }}
    .stButton>button {{ transition: all 0.2s ease-in-out; border: 1px solid #444444; }}
    .stButton>button:hover {{ border-color: #B3FF00 !important; color: #B3FF00 !important; box-shadow: 0 0 15px rgba(179, 255, 0, 0.7), 0 0 10px rgba(179, 255, 0, 0.5) !important; transform: translateY(-2px); }}
    .stTextInput>div>div>input:focus, .stTextInput>div>div>textarea:focus {{ border-color: #B3FF00 !important; box-shadow: 0 0 8px #B3FF00 !important; }}
    section[data-testid="stSidebar"] {{ background: rgba(13, 13, 13, 0.98) !important; backdrop-filter: blur(10px); border-right: 2px solid rgba(179, 255, 0, 0.3); }}
    section[data-testid="stSidebar"] > div {{ background: transparent !important; padding: 2rem 1rem !important; }}
    .role-card-container {{ position: relative; height: 250px; width: 100%; border-radius: 20px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); transition: transform 0.3s ease, box-shadow 0.3s ease; overflow: hidden; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5); margin-bottom: 20px; display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; color: white; cursor: pointer; }}
    .card-customer {{ border-bottom: 4px solid #00FFFF; }}
    .card-customer:hover {{ transform: translateY(-5px); box-shadow: 0 0 30px rgba(0, 255, 255, 0.7); border-color: #00FFFF; }}
    .card-business {{ border-bottom: 4px solid #FFD700; }}
    .card-business:hover {{ transform: translateY(-5px); box-shadow: 0 0 30px rgba(255, 215, 0, 0.7); border-color: #FFD700; }}
    .card-icon {{ font-size: 4rem; margin-bottom: 10px; text-shadow: 0 0 10px rgba(255, 255, 255, 0.5); }}
    .card-title {{ font-size: 1.5rem; font-weight: 700; margin-bottom: 5px; }}
    .card-desc {{ font-size: 0.9rem; opacity: 0.8; padding: 0 20px; }}
    .login-container {{ padding: 30px; border-radius: 10px; background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(8px); max-width: 100%; margin: 0 auto; box-shadow: 0 0 20px rgba(0, 0, 0, 0.5); }}
    
    /* Improved result box styling - smaller fonts */
    .element-container .stSuccess {{ 
        font-size: 0.75rem !important;
        line-height: 1.3 !important;
        padding: 0.6rem !important;
        border-radius: 8px !important;
        background: rgba(0, 255, 0, 0.1) !important;
        border-left: 3px solid #00ff00 !important;
    }}
    .element-container .stSuccess p {{ 
        font-size: 0.75rem !important;
        margin: 0 !important;
        font-family: 'Courier New', monospace !important;
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }}
    .element-container .stError {{ 
        font-size: 0.75rem !important;
        line-height: 1.3 !important;
        padding: 0.6rem !important;
        border-radius: 8px !important;
        background: rgba(255, 0, 0, 0.1) !important;
        border-left: 3px solid #ff0000 !important;
    }}
    .element-container .stError p {{ 
        font-size: 0.75rem !important;
        margin: 0 !important;
        font-family: 'Courier New', monospace !important;
        white-space: pre-wrap !important;
        word-break: break-word !important;
    }}
    
    /* Thought process panel styling - much smaller fonts */
    div[data-testid="column"]:last-child .element-container h3 {{
        font-size: 0.9rem !important;
        margin-bottom: 0.5rem !important;
    }}
    div[data-testid="column"]:last-child .element-container {{
        font-size: 0.7rem !important;
    }}
    div[data-testid="column"]:last-child .stExpander {{
        font-size: 0.7rem !important;
        margin-bottom: 0.3rem !important;
    }}
    div[data-testid="column"]:last-child .stExpander summary {{
        font-size: 0.7rem !important;
        padding: 0.3rem 0.5rem !important;
    }}
    div[data-testid="column"]:last-child .stExpander div[role="button"] {{
        font-size: 0.7rem !important;
    }}
    div[data-testid="column"]:last-child .stSuccess {{
        font-size: 0.65rem !important;
        padding: 0.4rem !important;
        line-height: 1.2 !important;
    }}
    div[data-testid="column"]:last-child .stSuccess p {{
        font-size: 0.65rem !important;
        margin: 0.2rem 0 !important;
    }}
    div[data-testid="column"]:last-child .stSuccess strong {{
        font-size: 0.65rem !important;
        font-weight: 600 !important;
    }}
    div[data-testid="column"]:last-child .stError {{
        font-size: 0.65rem !important;
        padding: 0.4rem !important;
        line-height: 1.2 !important;
    }}
    div[data-testid="column"]:last-child .stError p {{
        font-size: 0.65rem !important;
        margin: 0.2rem 0 !important;
    }}
    div[data-testid="column"]:last-child pre {{
        font-size: 0.65rem !important;
        padding: 0.3rem !important;
        margin: 0.2rem 0 !important;
    }}
    div[data-testid="column"]:last-child code {{
        font-size: 0.65rem !important;
    }}
    
    /* Additional styling for customer result text */
    div[data-testid="column"]:last-child .stSuccess div {{
        font-size: 0.65rem !important;
    }}
    div[data-testid="column"]:last-child .stSuccess * {{
        font-size: 0.65rem !important;
        line-height: 1.3 !important;
    }}
    
    /* Notification styles */
    .notification-container {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 320px;
        max-width: 400px;
        animation: slideInBottom 0.5s ease-out, fadeOut 1s ease-out 9s;
        animation-fill-mode: forwards;
        margin-bottom: 10px;
    }}
    
    @keyframes slideInBottom {{
        from {{
            transform: translateX(400px);
            opacity: 0;
        }}
        to {{
            transform: translateX(0);
            opacity: 1;
        }}
    }}
    
    @keyframes fadeOut {{
        from {{
            opacity: 1;
        }}
        to {{
            opacity: 0;
            transform: translateX(400px);
        }}
    }}
    
    .notification-box {{
        background: rgba(0, 0, 0, 0.95);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8);
        border: 2px solid;
        position: relative;
        backdrop-filter: blur(10px);
        margin-bottom: 10px;
    }}
    
    .notification-success {{
        border-color: #00ff00;
        box-shadow: 0 8px 32px rgba(0, 255, 0, 0.3);
    }}
    
    .notification-error {{
        border-color: #ff0000;
        box-shadow: 0 8px 32px rgba(255, 0, 0, 0.3);
    }}
    
    .notification-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
    }}
    
    .notification-title {{
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        margin: 0;
    }}
    
    .notification-close {{
        background: transparent;
        border: none;
        color: rgba(255, 255, 255, 0.6);
        font-size: 1.5rem;
        cursor: pointer;
        padding: 0;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: color 0.2s;
        line-height: 1;
    }}
    
    .notification-close:hover {{
        color: #ffffff;
    }}
    
    .notification-message {{
        color: rgba(255, 255, 255, 0.9);
        font-size: 0.9rem;
        line-height: 1.4;
        margin: 0;
    }}
    
    .notification-success .notification-title {{
        color: #00ff00;
    }}
    
    .notification-error .notification-title {{
        color: #ff0000;
    }}
    
    .notification-stack {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 9999;
        display: flex;
        flex-direction: column-reverse;
        gap: 10px;
    }}
    
    /* Welcome banner styling */
    .welcome-banner {{
        background: linear-gradient(135deg, rgba(179, 255, 0, 0.15) 0%, rgba(0, 255, 255, 0.15) 100%);
        border: 1px solid rgba(179, 255, 0, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(179, 255, 0, 0.2);
    }}
    .welcome-text {{
        font-size: 1.2rem;
        font-weight: 600;
        color: #B3FF00;
        margin: 0;
        text-shadow: 0 0 10px rgba(179, 255, 0, 0.5);
    }}
    .welcome-subtitle {{
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.8);
        margin: 0.3rem 0 0 0;
    }}
</style>
""", unsafe_allow_html=True)


# --- Core Logic: Conversational Loop ---

def handle_chat_interaction(prompt, role):
    if role == "customer":
        tools = REGISTERED_CUSTOMER_TOOLS
        history_key = "customer_history"
        system_instruction = get_customer_system_instruction()
        # Create function declarations for customer tools
        tool_declarations = [
            types.FunctionDeclaration(
                name='get_customer_orders',
                description='Returns ALL order information for a specific customer',
                parameters={
                    'type': 'object',
                    'properties': {
                        'customer_id': {
                            'type': 'string',
                            'description': 'The customer\'s unique ID (e.g., CUST_001)'
                        }
                    },
                    'required': ['customer_id']
                }
            )
        ]
    else: 
        tools = REGISTERED_BUSINESS_TOOLS_ALL 
        history_key = "business_history"
        system_instruction = SYSTEM_INSTRUCTION_BUSINESS
        # Create function declarations for business tools
        tool_declarations = [
            types.FunctionDeclaration(
                name='execute_pandas_code_business',
                description='Executes Python code for business analytics with access to ALL tables',
                parameters={
                    'type': 'object',
                    'properties': {
                        'python_code': {
                            'type': 'string',
                            'description': 'Python code to execute. Must assign result to variable named "result"'
                        }
                    },
                    'required': ['python_code']
                }
            )
        ]
    
    contents = [
        types.Content(
            role="model" if msg["role"] == "assistant" else msg["role"],
            parts=[types.Part.from_text(text=str(msg["content"]))]
        )
        for msg in st.session_state[history_key]
    ]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    # Create tool with function declarations
    tool = types.Tool(function_declarations=tool_declarations)
    
    tool_config = types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode='AUTO')
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        tools=[tool],
        tool_config=tool_config,
        temperature=0
    )
    
    # Create function map for automatic function calling
    function_map = {}
    for func in tools:
        function_map[func.__name__] = func
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=contents,
        config=config
    )
    
    # Handle function calling manually if needed
    current_tool_steps = []
    
    # Check if response has candidates and content
    if (hasattr(response, 'candidates') and 
        response.candidates and 
        len(response.candidates) > 0 and
        hasattr(response.candidates[0], 'content') and
        response.candidates[0].content and
        hasattr(response.candidates[0].content, 'parts') and
        response.candidates[0].content.parts):
        
        for part in response.candidates[0].content.parts:
            if hasattr(part, 'function_call') and part.function_call:
                func_call = part.function_call
                func_name = func_call.name
                func_args = dict(func_call.args)
                
                current_tool_steps.append({
                    "type": "Tool Call",
                    "name": func_name,
                    "args": func_args
                })
                
                # Execute the function
                if func_name in function_map:
                    try:
                        result = function_map[func_name](**func_args)
                        current_tool_steps.append({
                            "type": "Tool Output",
                            "output": result
                        })
                        
                        # Send the function response back to the model
                        function_response_content = types.Content(
                            role="user",
                            parts=[types.Part.from_function_response(
                                name=func_name,
                                response={'result': result}
                            )]
                        )
                        
                        contents.append(response.candidates[0].content)
                        contents.append(function_response_content)
                        
                        # Get final response
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=contents,
                            config=config
                        )
                    except Exception as e:
                        error_msg = f"Error executing {func_name}: {str(e)}"
                        current_tool_steps.append({
                            "type": "Tool Output",
                            "output": error_msg
                        })

    st.session_state.last_raw_response = str(response)
    
    # Extract text safely
    response_text = ""
    if hasattr(response, 'text'):
        response_text = response.text
    elif hasattr(response, 'candidates') and response.candidates:
        # Try to extract text from candidates
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and candidate.content:
                for part in candidate.content.parts:
                    if hasattr(part, 'text') and part.text:
                        response_text += part.text
    
    return response_text if response_text else "I apologize, but I couldn't generate a proper response. Please try again.", current_tool_steps


# --- Authentication Functions ---

def authenticate_user(username, password, expected_role):
    global CREDENTIALS
    CREDENTIALS = load_credentials()
    
    if username in CREDENTIALS:
        user_data = CREDENTIALS[username]
        if user_data["password"] == password:
            if user_data["role"] == expected_role:
                st.session_state.authenticated_user = username
                # Store customer_id for customer role
                if expected_role == "customer":
                    st.session_state.customer_id = user_data.get("customer_id")
                st.session_state.page = f"{expected_role}_chat"
                st.session_state.auth_role_pending = None 
                st.rerun()
            else:
                st.error("Authentication failed: Incorrect role for this user.")
        else:
            st.error("Authentication failed: Incorrect password.")
    else:
        st.error("Authentication failed: User ID not found.")

def create_new_user(new_username, new_password, role, customer_id=None):
    global CREDENTIALS
    CREDENTIALS = load_credentials()
    
    if not new_username or not new_password:
        st.warning("User ID and Password cannot be empty.")
        return

    if new_username in CREDENTIALS:
        st.error(f"Registration failed: User ID '{new_username}' already exists. Please choose a different name.")
    else:
        if len(new_password) < 6:
            st.error("Password must be at least 6 characters long.")
            return

        user_data = {
            "password": new_password,
            "role": role
        }
        
        # Add customer_id for customer accounts
        if role == "customer":
            if not customer_id:
                st.error("Customer ID is required for customer accounts.")
                return
            user_data["customer_id"] = customer_id.strip().upper()
        
        CREDENTIALS[new_username] = user_data
        
        if save_credentials(CREDENTIALS):
            st.success(f"Account for '{new_username}' created successfully! You can now log in.")
        else:
            st.error("Failed to save new user account.")


def render_auth_page():
    st.sidebar.empty()
    
    role = st.session_state.auth_role_pending
    display_role = "Customer" if role == "customer" else "Business Owner"
    
    st.markdown(f"<h1 style='text-align: center; margin-top: 80px; margin-bottom: 30px; font-size: 1.8rem;'>🔐 {display_role} Portal Access</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        
        st.markdown("<h3>Login</h3>", unsafe_allow_html=True)
        
        with st.form("login_form"):
            login_username = st.text_input("User ID", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_button = st.form_submit_button("Login", use_container_width=True)

            if login_button:
                authenticate_user(login_username, login_password, role)
        
        st.markdown("---")
        st.markdown("<h3>New User Registration</h3>", unsafe_allow_html=True)

        with st.form("register_form"):
            register_username = st.text_input("New User ID", key="register_username")
            register_password = st.text_input("New Password (Min 6 chars)", type="password", key="register_password")
            
            # Customer-specific field
            register_customer_id = None
            if role == "customer":
                register_customer_id = st.text_input("Your Customer ID (e.g., CUST_001)", key="register_customer_id", 
                                                     help="This links your account to your orders in our system")
            
            register_button = st.form_submit_button("Create Account", use_container_width=True, type="secondary")

            if register_button:
                create_new_user(register_username, register_password, role, register_customer_id)
            
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: center; margin-top: 30px;'>", unsafe_allow_html=True)
    col_back1, col_back2, col_back3 = st.columns([1, 2, 1])
    with col_back2:
        if st.button("⬅️ Cancel and Go Back", key="auth_back", use_container_width=True):
            st.session_state.auth_role_pending = None
            st.session_state.page = "selector"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    

def render_selector_page():
    st.sidebar.empty()
    # Reset notification flag when going back to selector
    st.session_state.notification_shown = False
    st.markdown("<div style='margin-top: 50px;'>", unsafe_allow_html=True) 
    st.markdown("<h1 style='text-align: center; margin-bottom: 40px;'>🤖 Autonomous Business Intelligence Agent</h1>", unsafe_allow_html=True)
    col_left, col_cust, col_mid, col_biz, col_right = st.columns([1, 3, 0.5, 3, 1])
    with col_cust:
        st.markdown("""<div class="role-card-container card-customer"><div class="card-icon">📦</div><div class="card-title">Customer Portal</div><div class="card-desc">Track Orders • Check Delivery • Shipping Updates</div></div>""", unsafe_allow_html=True)
        if st.button("Enter Customer Portal", use_container_width=True):
            st.session_state.auth_role_pending = "customer"
            st.session_state.page = "auth"
            st.rerun()
    with col_biz:
        st.markdown("""<div class="role-card-container card-business"><div class="card-icon">📊</div><div class="card-title">Business Command</div><div class="card-desc">Revenue Analytics • Anomaly Detection • Sales Data</div></div>""", unsafe_allow_html=True)
        if st.button("Enter Business Command", use_container_width=True):
            st.session_state.auth_role_pending = "business"
            st.session_state.page = "auth"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_chat_page(role):
    if st.session_state.authenticated_user is None:
        st.warning("Access Denied. Please login.")
        st.session_state.page = "selector"
        st.rerun()
        return

    # Show notification on first load
    if not st.session_state.notification_shown:
        if role == "customer" and st.session_state.customer_id:
            # Check customer order status
            status_info = check_customer_order_status(st.session_state.customer_id)
            notification_type = "success" if status_info["status"] != "delayed" else "error"
            notification_title = "Order Status" if status_info["status"] != "delayed" else "Delivery Alert"
            
            notification_html = f"""
            <style>
                .notification-stack {{
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    z-index: 9999;
                    display: flex;
                    flex-direction: column-reverse;
                    gap: 10px;
                    max-width: 400px;
                }}
                .notification-container {{
                    min-width: 320px;
                    animation: slideInBottom 0.5s ease-out;
                }}
                @keyframes slideInBottom {{
                    from {{
                        transform: translateX(400px);
                        opacity: 0;
                    }}
                    to {{
                        transform: translateX(0);
                        opacity: 1;
                    }}
                }}
                @keyframes fadeOut {{
                    from {{
                        opacity: 1;
                    }}
                    to {{
                        opacity: 0;
                        transform: translateX(400px);
                    }}
                }}
                .notification-box {{
                    background: rgba(0, 0, 0, 0.95);
                    border-radius: 12px;
                    padding: 1rem 1.2rem;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8);
                    border: 2px solid;
                    backdrop-filter: blur(10px);
                }}
                .notification-success {{
                    border-color: #00ff00;
                    box-shadow: 0 8px 32px rgba(0, 255, 0, 0.3);
                }}
                .notification-error {{
                    border-color: #ff0000;
                    box-shadow: 0 8px 32px rgba(255, 0, 0, 0.3);
                }}
                .notification-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 0.5rem;
                }}
                .notification-title {{
                    font-size: 1rem;
                    font-weight: 700;
                    color: #ffffff;
                    margin: 0;
                }}
                .notification-close {{
                    background: transparent;
                    border: none;
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 1.8rem;
                    cursor: pointer;
                    padding: 0;
                    width: 28px;
                    height: 28px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: color 0.2s;
                    line-height: 0.8;
                }}
                .notification-close:hover {{
                    color: #ffffff;
                }}
                .notification-message {{
                    color: rgba(255, 255, 255, 0.9);
                    font-size: 0.9rem;
                    line-height: 1.4;
                    margin: 0;
                    word-break: break-word;
                }}
                .notification-success .notification-title {{
                    color: #00ff00;
                }}
                .notification-error .notification-title {{
                    color: #ff0000;
                }}
            </style>
            <div class="notification-stack">
                <div class="notification-container" id="notif-customer">
                    <div class="notification-box notification-{notification_type}">
                        <div class="notification-header">
                            <h3 class="notification-title">{notification_title}</h3>
                            <button class="notification-close" onclick="document.getElementById('notif-customer').remove()">×</button>
                        </div>
                        <p class="notification-message">{status_info["message"]}</p>
                    </div>
                </div>
            </div>
            <script>
                setTimeout(function() {{
                    const container = document.getElementById('notif-customer');
                    if (container) {{
                        container.style.animation = 'fadeOut 1s ease-out forwards';
                        setTimeout(() => container.remove(), 1000);
                    }}
                }}, 10000);
            </script>
            """
            st.components.v1.html(notification_html, height=200)
            
        elif role == "business":
            # Check business status - need to run audit if pending
            if st.session_state.revenue_alert_status == "Pending (Run Audit)":
                st.session_state.revenue_alert_status = check_for_revenue_anomalies()
            if st.session_state.delay_alert_status == "Pending (Run Audit)":
                st.session_state.delay_alert_status = check_for_critical_delays()
            
            revenue_status = st.session_state.revenue_alert_status
            delay_status = st.session_state.delay_alert_status
            
            # Check for revenue issues
            has_revenue_issue = "CRITICAL" in revenue_status
            has_delay_issue = "ALERT" in delay_status
            
            notifications = []
            
            # Revenue notification
            if has_revenue_issue:
                # Extract amount from status
                import re
                amount_match = re.search(r'\$([0-9,]+\.[0-9]{2})', revenue_status)
                amount = amount_match.group(0) if amount_match else ""
                order_match = re.search(r'Order: ([A-Z_0-9]+)', revenue_status)
                order = order_match.group(1) if order_match else ""
                
                notifications.append({
                    "type": "error",
                    "title": "💰 Revenue Alert",
                    "message": f"⚠️ Anomaly: {amount} (Order: {order})"
                })
            else:
                notifications.append({
                    "type": "success",
                    "title": "💰 Revenue Status",
                    "message": "✅ Revenue metrics normal"
                })
            
            # Logistics notification
            if has_delay_issue:
                # Extract the number of delayed orders
                import re
                match = re.search(r'(\d+) orders', delay_status)
                count = match.group(1) if match else "Multiple"
                
                # Extract order IDs
                orders_match = re.search(r'Affected Orders: ([A-Z_0-9, ]+)', delay_status)
                orders = orders_match.group(1) if orders_match else ""
                
                notifications.append({
                    "type": "error",
                    "title": "🚚 Logistics Alert",
                    "message": f"⚠️ {count} orders delayed: {orders}"
                })
            else:
                notifications.append({
                    "type": "success",
                    "title": "🚚 Logistics Status",
                    "message": "✅ All deliveries on schedule"
                })
            
            # Build notification HTML
            notification_html = """
            <style>
                .notification-stack {
                    position: fixed;
                    bottom: 20px;
                    right: 20px;
                    z-index: 9999;
                    display: flex;
                    flex-direction: column-reverse;
                    gap: 10px;
                    max-width: 400px;
                }
                .notification-container {
                    min-width: 320px;
                    animation: slideInBottom 0.5s ease-out;
                }
                @keyframes slideInBottom {
                    from {
                        transform: translateX(400px);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
                @keyframes fadeOut {
                    from {
                        opacity: 1;
                    }
                    to {
                        opacity: 0;
                        transform: translateX(400px);
                    }
                }
                .notification-box {
                    background: rgba(0, 0, 0, 0.95);
                    border-radius: 12px;
                    padding: 1rem 1.2rem;
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8);
                    border: 2px solid;
                    backdrop-filter: blur(10px);
                }
                .notification-success {
                    border-color: #00ff00;
                    box-shadow: 0 8px 32px rgba(0, 255, 0, 0.3);
                }
                .notification-error {
                    border-color: #ff0000;
                    box-shadow: 0 8px 32px rgba(255, 0, 0, 0.3);
                }
                .notification-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 0.5rem;
                }
                .notification-title {
                    font-size: 1rem;
                    font-weight: 700;
                    color: #ffffff;
                    margin: 0;
                }
                .notification-close {
                    background: transparent;
                    border: none;
                    color: rgba(255, 255, 255, 0.6);
                    font-size: 1.8rem;
                    cursor: pointer;
                    padding: 0;
                    width: 28px;
                    height: 28px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    transition: color 0.2s;
                    line-height: 0.8;
                }
                .notification-close:hover {
                    color: #ffffff;
                }
                .notification-message {
                    color: rgba(255, 255, 255, 0.9);
                    font-size: 0.9rem;
                    line-height: 1.4;
                    margin: 0;
                    word-break: break-word;
                }
                .notification-success .notification-title {
                    color: #00ff00;
                }
                .notification-error .notification-title {
                    color: #ff0000;
                }
            </style>
            <div class="notification-stack">
            """
            
            for i, notif in enumerate(notifications):
                notification_html += f"""
                <div class="notification-container" id="notif-{i}">
                    <div class="notification-box notification-{notif['type']}">
                        <div class="notification-header">
                            <h3 class="notification-title">{notif['title']}</h3>
                            <button class="notification-close" onclick="document.getElementById('notif-{i}').remove()">×</button>
                        </div>
                        <p class="notification-message">{notif['message']}</p>
                    </div>
                </div>
                """
            
            notification_html += """
            </div>
            <script>
                setTimeout(function() {
                    const containers = document.querySelectorAll('.notification-container');
                    containers.forEach(c => {
                        c.style.animation = 'fadeOut 1s ease-out forwards';
                        setTimeout(() => c.remove(), 1000);
                    });
                }, 10000);
            </script>
            """
            
            st.components.v1.html(notification_html, height=250)
        
        st.session_state.notification_shown = True

    # --- SIDEBAR CONTENT ---
    with st.sidebar:
        st.header("Session")
        if role == "customer" and st.session_state.customer_id:
            st.info(f"**Customer:** {st.session_state.authenticated_user}\n**ID:** {st.session_state.customer_id}")
        else:
            st.info(f"**Logged in as:** {st.session_state.authenticated_user}")
            
        if st.button("🚪 Logout & Return Home", key="sidebar_logout", use_container_width=True): 
            st.session_state.authenticated_user = None 
            st.session_state.customer_id = None
            st.session_state.auth_role_pending = None
            st.session_state.customer_history = []
            st.session_state.business_history = []
            st.session_state.audit_log = []
            st.session_state.notification_shown = False
            st.session_state.page = "selector"
            st.rerun()
        st.markdown("---")
        if role == "business":
            st.header("⚡ System Actions")
            if st.button("🚨 Run System Audit", key="run_audit_sidebar", use_container_width=True):
                with st.spinner("Running diagnostic scans on Revenue & Logistics..."):
                    st.session_state.revenue_alert_status = check_for_revenue_anomalies()
                    st.session_state.delay_alert_status = check_for_critical_delays()
                    st.session_state.notification_shown = False  # Reset to show new notification
                st.rerun()
            st.markdown("---")
            st.header("📊 Live Status")
            if "CRITICAL" in st.session_state.revenue_alert_status:
                st.error(f"💰 **Revenue:**\n{st.session_state.revenue_alert_status}")
            else:
                st.success(f"💰 **Revenue:**\nNormal")
            if "ALERT" in st.session_state.delay_alert_status:
                st.error(f"🚚 **Logistics:**\n{st.session_state.delay_alert_status}")
            else:
                st.success(f"🚚 **Logistics:**\nNormal")

    # --- MAIN CHAT AREA ---
    st.markdown("<div style='margin-top: 30px;'>", unsafe_allow_html=True)
    
    # Welcome banner
    username = st.session_state.authenticated_user
    if role == "customer":
        greeting = f"Welcome back, {username}!"
        subtitle = "I'm here to help you track your orders and deliveries"
        st.markdown(f"""
        <div class='welcome-banner'>
            <p class='welcome-text'>👋 {greeting}</p>
            <p class='welcome-subtitle'>{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        greeting = f"Welcome, {username}!"
        subtitle = "Your intelligent business analytics assistant"
        st.markdown(f"""
        <div class='welcome-banner'>
            <p class='welcome-text'>📊 {greeting}</p>
            <p class='welcome-subtitle'>{subtitle}</p>
        </div>
        """, unsafe_allow_html=True)
    
    if role == "customer":
        st.title("💬 Customer Support")
        st.caption("How can I help you with your orders today?")
        history_key = "customer_history"
    else:
        st.title("📈 Business Intelligence Hub")
        st.caption("Ask anything about your data. I can analyze trends, totals, and cross-table insights.")
        history_key = "business_history"

    col_chat, col_glass = st.columns([0.65, 0.35], gap="large")

    with col_chat:
        for message in st.session_state[history_key]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Type your query here..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state[history_key].append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                final_text, tool_steps = handle_chat_interaction(prompt, role)
            
            st.session_state.audit_log = tool_steps
            st.session_state[history_key].append({"role": "assistant", "content": final_text})
            st.rerun()

    with col_glass:
        st.subheader("🔍 Agent Thought Process")
        if st.session_state.audit_log:
            for step in st.session_state.audit_log:
                if step["type"] == "Tool Call":
                    with st.expander(f"🛠️ Executing: {step['name']}", expanded=False):
                        if step['name'] == 'execute_pandas_code_business':
                             st.code(step['args'].get('python_code', ''), language='python')
                        else:
                             st.json(step['args'])
                elif step["type"] == "Tool Output":
                    res = str(step['output'])
                    if "ERROR" in res or "CRITICAL" in res:
                        st.error(f"**Result:** {res}")
                    else:
                        st.success(f"**Result:** {res}")
        else:
            st.markdown("<p style='opacity: 0.6; font-style: italic;'>No external tools required for this response.</p>", unsafe_allow_html=True)
        
        with st.expander("🛠️ Developer Debug"):
            if st.session_state.last_raw_response:
                st.write(st.session_state.last_raw_response)
                
    st.markdown("</div>", unsafe_allow_html=True)


# --- ROUTER ---

if st.session_state.page == "selector":
    render_selector_page()
elif st.session_state.page == "auth":
    render_auth_page()
elif st.session_state.page == "customer_chat":
    render_chat_page("customer")
elif st.session_state.page == "business_chat":
    render_chat_page("business")