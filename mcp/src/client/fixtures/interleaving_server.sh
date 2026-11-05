#!/bin/sh
# A scripted MCP server: after `tools/list` arrives it sends a notification, a
# ping and a sampling request before the tools/list response, and echoes each
# answer it gets back to the client as a `test/echo` notification.
read -r initialize
printf '%s\n' '{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05","capabilities":{},"serverInfo":{"name":"script"}}}'
read -r initialized
read -r list
printf '%s\n' '{"jsonrpc":"2.0","method":"notifications/tools/list_changed"}'
printf '%s\n' '{"jsonrpc":"2.0","id":"ping-1","method":"ping"}'
read -r pong
printf '{"jsonrpc":"2.0","method":"test/echo","params":%s}\n' "$pong"
printf '%s\n' '{"jsonrpc":"2.0","id":"sampling-1","method":"sampling/createMessage","params":{}}'
read -r rejection
printf '{"jsonrpc":"2.0","method":"test/echo","params":%s}\n' "$rejection"
printf '%s\n' '{"jsonrpc":"2.0","id":2,"result":{"tools":[]}}'
# Stay connected until the client hangs up.
while read -r line; do :; done
