# Scythe Context Engine MCP Server

An MCP (Model Context Protocol) server that provides semantic code search capabilities for your codebase using the Scythe Context Engine.

## What it does

This MCP server exposes a single tool called `query` that allows AI assistants to:

1. **Automatically index your project** - Performs incremental indexing of your codebase, only re-processing changed files
2. **Semantic search** - Uses embeddings and reranking to find the most relevant code context for any query
3. **Return refined context** - Provides formatted, relevant code snippets and information

## Installation

### Prerequisites

- Python 3.8+
- uv package manager (recommended) OR virtual environment

### Setup with uv (Recommended)

1. **Install dependencies:**
   ```bash
   uv pip install -e ..
   ```

2. **Verify installation:**
   ```bash
   uv run python server.py --help
   ```

### Setup with uv Virtual Environment

1. **Create uv virtual environment:**
   ```bash
   uv venv
   ```

2. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate  # On macOS/Linux
   # OR
   .venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   uv pip install -e ..
   ```

4. **Verify installation:**
   ```bash
   python server.py --help
   ```

### Setup with Standard Python Virtual Environment

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # OR
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -e ..
   ```

4. **Verify installation:**
   ```bash
   python server.py --help
   ```

## Usage

### Running the Server

Start the MCP server:

**With uv (recommended):**
```bash
uv run python server.py
```

**With uv venv:**
```bash
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
python server.py
```

**With standard venv:**
```bash
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
python server.py
```

The server will run and listen for MCP protocol messages.

### Using with Cursor AI

1. **Open Cursor Settings** (Cmd/Ctrl + ,)

2. **Navigate to MCP Settings:**
   - Go to `MCP` section in settings
   - Add to your MCP configuration:

   **With uv (recommended):**
   ```json
   {
     "mcpServers": {
       "scythe-context": {
         "command": "uv",
         "args": ["run", "python", "/absolute/path/to/mcp_server/server.py"],
         "cwd": "/absolute/path/to/Scythe-Context-Engine"
       }
     }
   }
   ```

   **With uv venv:**
   ```json
   {
     "mcpServers": {
       "scythe-context": {
         "command": "/absolute/path/to/Scythe-Context-Engine/.venv/bin/python",
         "args": ["/absolute/path/to/mcp_server/server.py"],
         "cwd": "/absolute/path/to/Scythe-Context-Engine"
       }
     }
   }
   ```

   **With standard venv:**
   ```json
   {
     "mcpServers": {
       "scythe-context": {
         "command": "/absolute/path/to/Scythe-Context-Engine/venv/bin/python",
         "args": ["/absolute/path/to/mcp_server/server.py"],
         "cwd": "/absolute/path/to/Scythe-Context-Engine"
       }
     }
   }
   ```

   Replace `/absolute/path/to` with the actual absolute paths to your Scythe Context Engine project.

3. **Save and restart Cursor**

### Using with VS Code

1. **Install the MCP extension:**
   - Install the "MCP (Model Context Protocol)" extension from the VS Code marketplace

2. **Configure MCP settings:**
   - Open VS Code settings (Cmd/Ctrl + ,)
   - Search for "MCP"
   - Add to your MCP configuration:

   **With uv (recommended):**
   ```json
   {
     "mcp": {
       "servers": {
         "scythe-context": {
           "command": "uv",
           "args": ["run", "python", "/absolute/path/to/mcp_server/server.py"],
           "cwd": "/absolute/path/to/Scythe-Context-Engine"
         }
       }
     }
   }
   ```

   **With uv venv:**
   ```json
   {
     "mcp": {
       "servers": {
         "scythe-context": {
           "command": "/absolute/path/to/Scythe-Context-Engine/.venv/bin/python",
           "args": ["/absolute/path/to/mcp_server/server.py"],
           "cwd": "/absolute/path/to/Scythe-Context-Engine"
         }
       }
     }
   }
   ```

   **With standard venv:**
   ```json
   {
     "mcp": {
       "servers": {
         "scythe-context": {
           "command": "/absolute/path/to/Scythe-Context-Engine/venv/bin/python",
           "args": ["/absolute/path/to/mcp_server/server.py"],
           "cwd": "/absolute/path/to/Scythe-Context-Engine"
         }
       }
     }
   }
   ```

   Replace `/absolute/path/to` with the actual absolute paths.

3. **Reload VS Code**

### Using with Claude Desktop

1. **Create/Edit Claude Desktop configuration:**
   - Location: `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS)
   - Location: `%APPDATA%\Claude\claude_desktop_config.json` (Windows)
   - Location: `~/.config/Claude/claude_desktop_config.json` (Linux)

2. **Add the server configuration:**

   **With uv (recommended):**
   ```json
   {
     "mcpServers": {
       "scythe-context": {
         "command": "uv",
         "args": ["run", "python", "/absolute/path/to/mcp_server/server.py"],
         "cwd": "/absolute/path/to/Scythe-Context-Engine"
       }
     }
   }
   ```

   **With uv venv:**
   ```json
   {
     "mcpServers": {
       "scythe-context": {
         "command": "/absolute/path/to/Scythe-Context-Engine/.venv/bin/python",
         "args": ["/absolute/path/to/mcp_server/server.py"],
         "cwd": "/absolute/path/to/Scythe-Context-Engine"
       }
     }
   }
   ```

   **With standard venv:**
   ```json
   {
     "mcpServers": {
       "scythe-context": {
         "command": "/absolute/path/to/Scythe-Context-Engine/venv/bin/python",
         "args": ["/absolute/path/to/mcp_server/server.py"],
         "cwd": "/absolute/path/to/Scythe-Context-Engine"
       }
     }
   }
   ```

3. **Restart Claude Desktop**

## The Query Tool

### Description

The `query` tool searches your codebase for relevant context and returns formatted results.

### Parameters

- **query** (string): Your search query or question about the codebase
- **project_location** (string): Absolute path to the project directory to search

### Example Usage

In Cursor AI or VS Code, you can ask questions like:

- "How does user authentication work?"
- "Find the database connection code"
- "Show me the error handling patterns"
- "Where is the payment processing logic?"

The tool will:
1. Index your project (incrementally - only changed files)
2. Search for semantically similar code
3. Rerank results using AI
4. Return the most relevant context

### Index Storage

- Indexes are stored in `.scythe_index/` folder within each project
- Incremental indexing prevents long query times on subsequent searches
- Safe to delete the `.scythe_index` folder to force a full re-index

## Troubleshooting

### Common Issues

1. **"Module not found" errors:**
   - Make sure you're running from the project root
   - Verify all dependencies are installed: `uv pip install -e .`

2. **Path issues:**
   - Always use absolute paths for `project_location`
   - Ensure the MCP server can access the target project directory

3. **Indexing takes too long:**
   - First run indexes the entire project
   - Subsequent runs are incremental (much faster)
   - You can delete `.scythe_index` to force full re-index

4. **Server won't start:**
   - Check that `uv` is installed and available
   - Verify Python path and dependencies
   - Check console/logs for error messages

### Debug Mode

To see server logs, run with verbose output:

```bash
uv run python server.py 2>&1
```

## Architecture

- **Indexer**: Processes source code into searchable chunks with embeddings
- **Query Engine**: Uses FAISS for fast semantic search
- **Reranker**: Uses LLM to improve result relevance
- **MCP Protocol**: Standardizes communication with AI assistants

## Contributing

This MCP server wraps the Scythe Context Engine. For improvements to the core search functionality, see the main project README.