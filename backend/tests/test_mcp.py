#Test 1
from mcp.server.fastmcp import FastMCP

print("Import successful")

mcp = FastMCP("test")

print("MCP instance created")

# Test 2
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("test")

@mcp.tool()
def hello(name: str) -> str:
    return f"Hello {name}"

print("Tool registered successfully")