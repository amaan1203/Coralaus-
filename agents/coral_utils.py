"""
Coral CLI wrapper utilities.
Executes Coral SQL queries via subprocess and tracks query count for demo purposes.
"""

import subprocess
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class CoralClient:
    """Wrapper around the Coral CLI for executing SQL queries."""

    def __init__(self):
        self.query_count = 0
        self.coral_path = "coral"
        self._verify_installation()

    def _verify_installation(self):
        """Check that coral CLI is available."""
        self.available = False
        paths_to_test = ["coral", "/opt/homebrew/bin/coral", "/usr/local/bin/coral"]
        for p in paths_to_test:
            try:
                result = subprocess.run(
                    [p, "--version"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    self.coral_path = p
                    self.available = True
                    logger.info(f"Coral CLI found at '{p}': {result.stdout.strip()}")
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        if not self.available:
            logger.warning("Coral CLI not installed or not working. Some features will use fallback mode.")

    def sql(self, query: str, timeout: int = 60) -> Optional[dict]:
        """
        Execute a Coral SQL query and return parsed JSON results.

        Args:
            query: SQL query string
            timeout: Max seconds to wait for response

        Returns:
            Parsed JSON result or None on failure
        """
        if not self.available:
            logger.error("Coral CLI not available. Cannot execute query.")
            return None

        self.query_count += 1
        logger.info(f"[Coral Query #{self.query_count}] {query[:120]}...")

        try:
            result = subprocess.run(
                [self.coral_path, "sql", "--format", "json", query],
                capture_output=True, text=True, timeout=timeout
            )

            if result.returncode != 0:
                logger.error(f"Coral query failed: {result.stderr}")
                return None

            output = result.stdout.strip()
            if not output:
                return {"results": []}

            parsed = json.loads(output)
            if isinstance(parsed, list):
                return {"results": parsed}
            elif isinstance(parsed, dict) and "results" in parsed:
                return parsed
            else:
                return {"results": [parsed]}

        except subprocess.TimeoutExpired:
            logger.error(f"Coral query timed out after {timeout}s")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Coral output as JSON: {e}")
            # Return raw output wrapped in a dict
            return {"raw": result.stdout.strip()}

    def sql_raw(self, query: str, timeout: int = 60) -> Optional[str]:
        """Execute a Coral SQL query and return raw string output."""
        if not self.available:
            return None

        self.query_count += 1
        logger.info(f"[Coral Query #{self.query_count}] {query[:120]}...")

        try:
            result = subprocess.run(
                [self.coral_path, "sql", query],
                capture_output=True, text=True, timeout=timeout
            )
            if result.returncode != 0:
                logger.error(f"Coral query failed: {result.stderr}")
                return None
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            logger.error(f"Coral query timed out after {timeout}s")
            return None

    def add_source(self, source_type: str, **kwargs) -> bool:
        """Register a Coral data source."""
        cmd = [self.coral_path, "source", "add", source_type]
        for key, value in kwargs.items():
            cmd.extend([f"--{key}", str(value)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logger.info(f"Source '{source_type}' added successfully")
                return True
            else:
                logger.error(f"Failed to add source: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Error adding source: {e}")
            return False

    def get_query_count(self) -> int:
        """Return the total number of Coral queries executed."""
        return self.query_count

    def reset_query_count(self):
        """Reset query counter (e.g. for a new paper analysis)."""
        self.query_count = 0


# Singleton instance
_client = None

def get_coral_client() -> CoralClient:
    """Get or create the global Coral client instance."""
    global _client
    if _client is None:
        _client = CoralClient()
    return _client
