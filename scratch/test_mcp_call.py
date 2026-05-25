import asyncio
import os
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import mcp_paperswithcode.server

async def main():
    server_path = mcp_paperswithcode.server.__file__
    
    server_params = StdioServerParameters(
        command="/Library/Frameworks/Python.framework/Versions/3.12/bin/mcp",
        args=["run", server_path],
        env=None
    )
    print("Connecting to MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Successfully initialized session!")
            
            print("Calling search_papers...")
            res = await session.call_tool("search_papers", arguments={"title": "Attention Is All You Need"})
            print(f"Result type: {type(res)}")
            print(f"Result: {res}")
            
            # Let's inspect the content list
            if hasattr(res, "content") and res.content:
                text_content = res.content[0].text
                print("Text content:")
                print(text_content[:1000])

if __name__ == "__main__":
    asyncio.run(main())
