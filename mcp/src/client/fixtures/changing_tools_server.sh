#!/bin/sh
# A scripted MCP server whose tool list changes mid-connection: it offers
# `alpha` and `beta`, and after the client's first `tools/call` it answers,
# sends `notifications/tools/list_changed` and from then on offers `beta` and
# `gamma`.
read -r initialize
printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{"tools":{"listChanged":true}},"serverInfo":{"name":"script"}}}'
read -r initialized
read -r list
printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"alpha","inputSchema":{"type":"object"}},{"name":"beta","inputSchema":{"type":"object"}}]}}'
read -r call
printf '%s\n' '{"jsonrpc":"2.0","id":3,"result":{"content":[],"isError":false}}'
printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}'
read -r relist
printf '%s\n' '{"jsonrpc":"2.0","id":4,"result":{"tools":[{"name":"beta","inputSchema":{"type":"object"}},{"name":"gamma","inputSchema":{"type":"object"}}]}}'
# Stay connected until the client hangs up.
while read -r line; do :; done
