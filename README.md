<!--
  ~ Copyright (c) 2023-2024 Datalayer, Inc.
  ~
  ~ BSD 3-Clause License
-->

**This folk is to refactor the original implementation to get rid of docker dependency.**

[![Datalayer](https://assets.datalayer.tech/datalayer-25.svg)](https://datalayer.io)

[![Become a Sponsor](https://img.shields.io/static/v1?label=Become%20a%20Sponsor&message=%E2%9D%A4&logo=GitHub&style=flat&color=1ABC9C)](https://github.com/sponsors/datalayer)

# 🪐 ✨ Jupyter MCP Server

[![Github Actions Status](https://github.com/datalayer/jupyter-mcp-server/workflows/Build/badge.svg)](https://github.com/datalayer/jupyter-mcp-server/actions/workflows/build.yml)
[![PyPI - Version](https://img.shields.io/pypi/v/jupyter-mcp-server)](https://pypi.org/project/jupyter-mcp-server)
[![smithery badge](https://smithery.ai/badge/@datalayer/jupyter-mcp-server)](https://smithery.ai/server/@datalayer/jupyter-mcp-server)

Jupyter MCP Server is a [Model Context Protocol](https://modelcontextprotocol.io/introduction) (MCP) server implementation that provides interaction with Jupyter notebooks 📓 running in a local JupyterLab 💻.

![Jupyter MCP Server](https://assets.datalayer.tech/jupyter-mcp/jupyter-mcp-server-claude-demo.gif)

## Start JupyterLab

Make sure you have the following installed. The modifications made on the notebook can be seen thanks to [Jupyter Real Time Collaboration](https://jupyterlab.readthedocs.io/en/stable/user/rtc.html) (RTC).

```bash
pip install jupyterlab jupyter-collaboration ipykernel
```

Then, start JupyterLab with the following command:

```bash
jupyter lab --port 8888 --IdentityProvider.token MY_TOKEN
```

## Install the server

```bash
# in case of debugging
pip uninstall jupyter-mcp-server jupyter_mcp_server
pip install -e .
```

## Usage with Claude Desktop

To use this with Claude Desktop, firstly modify `run_jupyter_mcp.sh` as you may require. 
Then make it executable

```bash
chmod +x run_jupyter_mcp.sh
```

> [!IMPORTANT]
> Ensure the port of the `SERVER_URL`and `TOKEN` match those used in the `jupyter lab` command.
> The `NOTEBOOK_PATH` should be relative to the directory where JupyterLab was started.

Then configure your claude_desktop_config.json to use it:


### MacOS

```json
{
  "mcpServers": {
    "jupyter": {
      "command": "/full/path/to/run_jupyter_mcp.sh"
    }
  }
}
```


## Components

### Tools

The server currently offers 3 tools:

1. `add_execute_code_cell`

- Add and execute a code cell in a Jupyter notebook.
- Input:
  - `cell_content`(string): Code to be executed.
- Returns: Cell output.

2. `add_markdown_cell`

- Add a markdown cell in a Jupyter notebook.
- Input:
  - `cell_content`(string): Markdown content.
- Returns: Success message.

3. `download_earth_data_granules`

   ⚠️ We plan to migrate this tool to a separate repository in the future as it is specific to Geospatial analysis.

- Add a code cell in a Jupyter notebook to download Earth data granules from NASA Earth Data.
- Input:
  - `folder_name`(string): Local folder name to save the data.
  - `short_name`(string): Short name of the Earth dataset to download.
  - `count`(int): Number of data granules to download.
  - `temporal` (tuple): (Optional) Temporal range in the format (date_from, date_to).
  - `bounding_box` (tuple): (Optional) Bounding box in the format (lower_left_lon, lower_left_lat, upper_right_lon, upper_right_lat).
- Returns: Cell output.
