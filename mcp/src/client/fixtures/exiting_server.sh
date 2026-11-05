#!/bin/sh
# A scripted MCP server that exits as soon as it has read a `tools/list`
# request, without answering it.
read -r initialize
printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"script"}}}'
read -r initialized
read -r list
exit 3
