# Copyright (c) 2023-2024 Datalayer, Inc.
#
# BSD 3-Clause License

import logging
import os
import argparse

from jupyter_kernel_client import KernelClient
from jupyter_nbmodel_client import NbModelClient, get_jupyter_notebook_websocket_url
from mcp.server.fastmcp import FastMCP
from rich.console import Console
from rich.logging import RichHandler

# Initialize FastMCP server
mcp = FastMCP("notebook")

# Setup logging
handlers = []
handlers.append(RichHandler(console=Console(stderr=True), rich_tracebacks=True))
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=handlers,
)

logger = logging.getLogger(__name__)


def extract_output(output: dict) -> str:
    """Extract output from a Jupyter notebook cell.

    Args:
        output: Output dictionary

    Returns:
        str: Output text
    """
    if output["output_type"] == "display_data":
        return output["data"]["text/plain"]
    elif output["output_type"] == "execute_result":
        return output["data"]["text/plain"]
    elif output["output_type"] == "stream":
        return output["text"]
    elif output["output_type"] == "error":
        return output["traceback"]
    else:
        return ""


@mcp.tool()
def add_execute_code_cell(cell_content: str) -> str:
    """Add and execute a code cell in a Jupyter notebook.

    Args:
        cell_content: Code content

    Returns:
        str: Cell output
    """
    # Get settings from environment or globals
    server_url = globals().get("SERVER_URL", os.getenv("SERVER_URL", "http://localhost:8888"))
    token = globals().get("TOKEN", os.getenv("TOKEN", ""))
    notebook_path = globals().get("NOTEBOOK_PATH", os.getenv("NOTEBOOK_PATH", "notebook.ipynb"))
    kernel = globals().get("kernel")

    logger.info(f"Adding and executing code cell in {notebook_path}")

    notebook = NbModelClient(
        get_jupyter_notebook_websocket_url(server_url=server_url, token=token, path=notebook_path)
    )
    notebook.start()

    cell_index = notebook.add_code_cell(cell_content)
    notebook.execute_cell(cell_index, kernel)

    ydoc = notebook._doc
    outputs = ydoc._ycells[cell_index]["outputs"]
    if len(outputs) == 0:
        cell_output = ""
    else:
        cell_output = [extract_output(output) for output in outputs]

    notebook.stop()

    return cell_output


@mcp.tool()
def add_markdown_cell(cell_content: str) -> str:
    """Add a markdown cell in a Jupyter notebook.

    Args:
        cell_content: Markdown content

    Returns:
        str: Success message
    """
    # Get settings from environment or globals
    server_url = globals().get("SERVER_URL", os.getenv("SERVER_URL", "http://localhost:8888"))
    token = globals().get("TOKEN", os.getenv("TOKEN", ""))
    notebook_path = globals().get("NOTEBOOK_PATH", os.getenv("NOTEBOOK_PATH", "notebook.ipynb"))

    logger.info(f"Adding markdown cell to {notebook_path}")

    notebook = NbModelClient(
        get_jupyter_notebook_websocket_url(server_url=server_url, token=token, path=notebook_path)
    )
    notebook.start()
    notebook.add_markdown_cell(cell_content)
    notebook.stop()

    return "Markdown cell added"


@mcp.tool()
def download_earth_data_granules(
        folder_name: str, short_name: str, count: int, temporal: tuple = None, bounding_box: tuple = None
) -> str:
    """Add a code cell in a Jupyter notebook to download Earth data granules from NASA Earth Data.

    Args:
        folder_name: Local folder name to save the data.
        short_name: Short name of the Earth dataset to download.
        count: Number of data granules to download.
        temporal: (Optional) Temporal range in the format (date_from, date_to).
        bounding_box: (Optional) Bounding box in the format (lower_left_lon, lower_left_lat,
        upper_right_lon, upper_right_lat).

    Returns:
        str: Cell output
    """
    # Get settings from environment or globals
    server_url = globals().get("SERVER_URL", os.getenv("SERVER_URL", "http://localhost:8888"))
    token = globals().get("TOKEN", os.getenv("TOKEN", ""))
    notebook_path = globals().get("NOTEBOOK_PATH", os.getenv("NOTEBOOK_PATH", "notebook.ipynb"))
    kernel = globals().get("kernel")

    logger.info(f"Downloading Earth data granules to {folder_name}")

    search_params = {"short_name": short_name, "count": count, "cloud_hosted": True}

    if temporal and len(temporal) == 2:
        search_params["temporal"] = temporal
    if bounding_box and len(bounding_box) == 4:
        search_params["bounding_box"] = bounding_box

    cell_content = f"""import earthaccess
earthaccess.login()

search_params = {search_params}  # Pass dictionary as a variable
results = earthaccess.search_data(**search_params)
files = earthaccess.download(results, "./{folder_name}")"""

    notebook = NbModelClient(
        get_jupyter_notebook_websocket_url(server_url=server_url, token=token, path=notebook_path)
    )
    notebook.start()

    cell_index = notebook.add_code_cell(cell_content)
    notebook.execute_cell(cell_index, kernel)

    notebook.stop()

    return f"Data downloaded in folder {folder_name}"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Jupyter MCP Server")
    parser.add_argument("--server-url", dest="server_url", type=str,
                        default=os.getenv("SERVER_URL", "http://localhost:8888"),
                        help="JupyterLab server URL")
    parser.add_argument("--token", type=str,
                        default=os.getenv("TOKEN", ""),
                        help="JupyterLab server token")
    parser.add_argument("--notebook-path", dest="notebook_path", type=str,
                        default=os.getenv("NOTEBOOK_PATH", "notebook.ipynb"),
                        help="Path to the notebook relative to JupyterLab root")
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Store settings as globals for tools to access
    global SERVER_URL, TOKEN, NOTEBOOK_PATH, kernel
    SERVER_URL = args.server_url
    TOKEN = args.token
    NOTEBOOK_PATH = args.notebook_path

    logger.info(f"Starting Jupyter MCP Server")
    logger.info(f"Server URL: {SERVER_URL}")
    logger.info(f"Notebook path: {NOTEBOOK_PATH}")

    # Initialize kernel client
    kernel = KernelClient(server_url=SERVER_URL, token=TOKEN)
    kernel.start()

    # Run the server
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()