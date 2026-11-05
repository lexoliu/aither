#!/bin/sh
# A scripted stdio MCP server offering one tool, `ping`, which answers `pong`.
read -r initialize
printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"ping-server"}}}'
read -r initialized
read -r list
printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"ping","description":"Answers pong","inputSchema":{"type":"object"}}]}}'
while read -r call; do
  printf '%s\n' '{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"pong"}],"isError":false}}'
done
