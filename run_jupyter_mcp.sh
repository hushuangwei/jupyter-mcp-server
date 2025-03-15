#!/bin/bash
export SERVER_URL="http://localhost:8889"
export TOKEN="test"
export NOTEBOOK_PATH="test.ipynb"
/opt/anaconda3/bin/python -m jupyter_mcp_server
