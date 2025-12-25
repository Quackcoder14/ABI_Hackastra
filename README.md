🤖 ABI Agent: Autonomous Business Intelligence Ecosystem

The ABI Agent is a sophisticated dual-sided platform that bridges the gap between complex business data and consumer satisfaction. Using Gemini 1.5 Flash and a custom function-calling architecture, it provides proactive diagnostics for business owners and intelligent logistics support for customers.

🌟 Core Architecture

The system is split into two specialized interfaces:

Business Command: For data-driven decision-making, trend analysis, and automated system auditing.

Customer Portal: For real-time, empathetic order tracking and delivery resolution.

The "Glass Box" feature provides full transparency by displaying the Agent's internal thought process and tool execution logs in real-time.

🚀 Installation & Setup

1. Clone the Repository

git clone https://github.com/QuackCoder14/ABI_Hackastra.git

cd ABI_Hackastra

2. Install Required Libraries
Ensure you have Python 3.10 or higher installed. Run:

pip install streamlit google-genai python-dotenv pandas

3. Configure Environment Variables (.env)
Create a file named .env in the root directory:

GEMINI_API_KEY=your_google_api_key_here

4. Initialize Credentials (credentials.json)
The application manages user access through a local JSON file.

Note: New users can also be registered directly through the app UI.

💻 Project Structure

app.py: The main Streamlit entry point. It handles the UI, session state, routing between portals, and the conversational loop.

tools.py: The "engine" of the agent. Contains the Python functions that interact with csv files and perform data analysis.

.env: Stores sensitive API configurations.

credentials.json: Stores user authentication and role data.

🏃 How to Run

#view orders.csv and take a note of the respective orders of customers.
Launch the application using Streamlit:

streamlit run app.py

Select Role: Choose Customer or Business Owner.

Login: Use the credentials defined in your JSON file.

Chat: 
Business: Ask "What is the status of my order ?"

Customer: Ask "What is the status of my order"

Audit: Watch the Thought Process pane to see the Agent selecting and executing tools.

📝 License
Distributed under the MIT License.
