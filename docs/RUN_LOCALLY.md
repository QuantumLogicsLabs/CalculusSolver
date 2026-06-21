# Running CalculusSolver Locally

This guide walks you through running the CalculusSolver backend locally, using Groq as the primary model.

## Prerequisites
- **Python 3.10+**
- **Git**
- A **Groq API Key** (You can obtain one from the [Groq Console](https://console.groq.com/keys))

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/QuantumLogicsLabs/CalculusSolver.git
   cd CalculusSolver
   ```

2. **Create and activate a virtual environment**
   ```bash
   # On macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate

   # On Windows
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies**
   Install the required packages. We use the `groq` library for fast model inference.
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   The application requires a Groq API key to process complex calculus requests intelligently.
   Create a `.env` file in the root of the project or export the variable in your shell:
   ```bash
   # Linux/macOS
   export GROQ_API_KEY="your-groq-api-key-here"

   # Windows (Command Prompt)
   set GROQ_API_KEY=your-groq-api-key-here

   # Windows (PowerShell)
   $env:GROQ_API_KEY="your-groq-api-key-here"
   ```

5. **Run the local development server**
   Use `uvicorn` to run the FastAPI / Starlette application:
   ```bash
   uvicorn api.app:app --reload
   ```

6. **Test the API**
   Once the server is running (usually at `http://127.0.0.1:8000`), you can test the `/api/solve` endpoint:
   ```bash
   curl -X POST http://127.0.0.1:8000/api/solve \
     -H "Content-Type: application/json" \
     -d '{"input": {"op": "diff", "var": "x", "expr": {"numi": {"terms": [{"coeff": 3, "var": {"x": 2}}]}, "deno": 1}}}'
   ```

## Deploying to Vercel

If you want to deploy this backend to Vercel:
1. Ensure your project is pushed to GitHub.
2. Go to the [Vercel Dashboard](https://vercel.com/) and create a new project from your repository.
3. In the project settings, under **Environment Variables**, add:
   - Key: `GROQ_API_KEY`
   - Value: `<your-groq-api-key>`
4. Deploy the project. Vercel will automatically use `vercel.json` to build and route the Python serverless functions.
