"""
PapersWithCode MCP Client wrapper.
Calls the mcp-paperswithcode server tools via subprocess (stdio transport).
Provides a clean Python API for paper search and repo discovery.
"""

import subprocess
import json
import logging
import os
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


class PapersWithCodeMCPClient:
    """
    Client for the @hbg/mcp-paperswithcode MCP server.
    Uses the MCP server via subprocess connection (stdio transport),
    with direct HTTP requests to PapersWithCode API as fallback.
    """

    PWC_API_BASE = "https://paperswithcode.com/api/v1"

    def __init__(self):
        self.call_count = 0
        self.use_mcp_server = False
        self.available = True
        self._check_mcp_availability()

    def _check_mcp_availability(self):
        """Check if PapersWithCode MCP server is installed and available."""
        try:
            import mcp_paperswithcode.server
            self.use_mcp_server = True
            logger.info("PapersWithCode MCP server is available for tool calls.")
        except ImportError:
            self.use_mcp_server = False
            logger.info("PapersWithCode MCP server package not found. Using HTTP fallback.")

    async def _call_mcp_tool_async(self, tool_name: str, arguments: dict) -> Optional[dict]:
        """Connect to the MCP server and run a tool asynchronously."""
        import mcp_paperswithcode.server
        import asyncio
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        server_path = mcp_paperswithcode.server.__file__
        
        # Look for mcp command path
        mcp_path = "/Library/Frameworks/Python.framework/Versions/3.12/bin/mcp"
        if not os.path.exists(mcp_path):
            mcp_path = "mcp"
            
        server_params = StdioServerParameters(
            command=mcp_path,
            args=["run", server_path],
            env=None
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                res = await session.call_tool(tool_name, arguments=arguments)
                if hasattr(res, "isError") and res.isError:
                    logger.warning(f"MCP server tool execution error: {res.content}")
                    return None
                if hasattr(res, "content") and res.content:
                    text_data = res.content[0].text
                    try:
                        return json.loads(text_data)
                    except json.JSONDecodeError:
                        return {"raw": text_data}
                return None

    def _call_mcp_tool(self, tool_name: str, arguments: dict) -> Optional[dict]:
        """Synchronous wrapper for MCP tool call."""
        if not self.use_mcp_server:
            return None
        self.call_count += 1
        logger.info(f"[PWC MCP #{self.call_count}] Calling tool '{tool_name}' via MCP server")
        try:
            return asyncio.run(self._call_mcp_tool_async(tool_name, arguments))
        except Exception as e:
            logger.warning(f"MCP server tool call failed: {e}. Falling back to HTTP.")
            return None

    def search_papers(self, query: str, items_per_page: int = 5) -> list[dict]:
        """Search for papers by title or keyword."""
        # Try MCP server first
        if self.use_mcp_server:
            res = self._call_mcp_tool("search_papers", {"title": query})
            if res and isinstance(res, dict) and "results" in res:
                return res["results"]

        # Fallback to direct HTTP
        import requests
        self.call_count += 1
        logger.info(f"[PWC HTTP #{self.call_count}] Searching papers: '{query[:80]}'")

        try:
            resp = requests.get(
                f"{self.PWC_API_BASE}/papers/",
                params={"q": query, "items_per_page": items_per_page},
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                logger.info(f"  Found {len(results)} papers")
                return results
            else:
                logger.error(f"  PWC API error: {resp.status_code}")
                return []
        except Exception as e:
            logger.error(f"  PWC search failed: {e}")
            return []

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[dict]:
        """Look up a specific paper by its arXiv ID."""
        # Try MCP server first
        if self.use_mcp_server:
            res = self._call_mcp_tool("search_papers", {"arxiv_id": arxiv_id})
            if res and isinstance(res, dict) and "results" in res and res["results"]:
                return res["results"][0]

        # Fallback to direct HTTP
        import requests
        self.call_count += 1
        logger.info(f"[PWC HTTP #{self.call_count}] Looking up arxiv:{arxiv_id}")

        try:
            resp = requests.get(
                f"{self.PWC_API_BASE}/papers/",
                params={"arxiv_id": arxiv_id},
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    logger.info(f"  Found paper: {results[0].get('title', 'Unknown')}")
                    return results[0]
            return None
        except Exception as e:
            logger.error(f"  PWC arxiv lookup failed: {e}")
            return None

    def get_paper_repositories(self, paper_id: str) -> list[dict]:
        """Get code repositories linked to a paper."""
        # Try MCP server first
        if self.use_mcp_server:
            res = self._call_mcp_tool("list_paper_repositories", {"paper_id": paper_id})
            if res and isinstance(res, dict) and "results" in res:
                return res["results"]

        # Fallback to direct HTTP
        import requests
        self.call_count += 1
        logger.info(f"[PWC HTTP #{self.call_count}] Getting repos for paper: {paper_id}")

        try:
            resp = requests.get(
                f"{self.PWC_API_BASE}/papers/{paper_id}/repositories/",
                timeout=15
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                logger.info(f"  Found {len(results)} repositories")
                return results
            else:
                logger.error(f"  PWC repos API error: {resp.status_code}")
                return []
        except Exception as e:
            logger.error(f"  PWC repos fetch failed: {e}")
            return []

    def find_implementation(self, title: str, arxiv_id: str = None) -> Optional[dict]:
        """
        High-level: find the best implementation for a paper.
        Tries title search first, then arxiv_id fallback.

        Args:
            title: Paper title
            arxiv_id: Optional arXiv ID for more precise lookup

        Returns:
            Best repo dict or None
        """
        # Strategy 1: Search by arXiv ID (most precise)
        if arxiv_id:
            paper = self.get_paper_by_arxiv_id(arxiv_id)
            if paper:
                paper_id = paper.get("id")
                if paper_id:
                    repos = self.get_paper_repositories(paper_id)
                    best = self._pick_best_repo(repos)
                    if best:
                        return best

        # Strategy 2: Search by title
        papers = self.search_papers(title, items_per_page=3)
        for paper in papers:
            paper_id = paper.get("id")
            if paper_id:
                repos = self.get_paper_repositories(paper_id)
                best = self._pick_best_repo(repos)
                if best:
                    return best

        logger.info("No implementation found via PapersWithCode")
        return None

    def _pick_best_repo(self, repos: list[dict]) -> Optional[dict]:
        """Pick the best repo: prefer official, then most stars."""
        if not repos:
            return None

        # Prefer official implementations
        official = [r for r in repos if r.get("is_official")]
        if official:
            return max(official, key=lambda r: r.get("stars", 0))

        # Otherwise pick by stars
        return max(repos, key=lambda r: r.get("stars", 0))

    def get_call_count(self) -> int:
        """Return total API calls made."""
        return self.call_count

    def reset_call_count(self):
        """Reset the call counter."""
        self.call_count = 0


# Singleton
_client = None

def get_pwc_client() -> PapersWithCodeMCPClient:
    """Get or create the global PapersWithCode client."""
    global _client
    if _client is None:
        _client = PapersWithCodeMCPClient()
    return _client
