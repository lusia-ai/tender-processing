"""Compatibility wrapper for the agent CLI and core functions."""
from pdf_agent.core.agent import build_graph, run_agent
from pdf_agent.cli.agent import main

__all__ = ["run_agent", "build_graph", "main"]


if __name__ == "__main__":
    main()
