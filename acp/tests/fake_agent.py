#!/usr/bin/env python3
"""Fake ACP agent over stdio for aither-acp tests.

Answers initialize / session/new / session/load / session/set_mode /
session/set_config_option. On session/prompt it streams an
agent_message_chunk, a tool_call with two tool_call_update updates, sends one
session/request_permission request, and answers the prompt once the client
responds — or answers `cancelled` when session/cancel arrives first. A prompt
whose text is "wait" only waits; "die" exits the process mid-turn.
"""

import json
import sys


def send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def respond(request_id, result):
    send({"jsonrpc": "2.0", "id": request_id, "result": result})


def update(session_id, update_payload):
    send(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"sessionId": session_id, "update": update_payload},
        }
    )


pending_prompt = None
session_id = ""
answered = {"perm": False, "fs": False}

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)
    except ValueError:
        continue

    if "method" in msg and "id" in msg:
        method, request_id, params = msg["method"], msg["id"], msg.get("params") or {}
        if method == "initialize":
            respond(
                request_id,
                {
                    "protocolVersion": 1,
                    "agentCapabilities": {
                        "loadSession": True,
                        "sessionCapabilities": {"list": {}, "delete": {}},
                    },
                    "agentInfo": {"name": "fake-agent-py", "version": "0.1.0"},
                },
            )
        elif method == "session/new":
            session_id = "sess-1"
            respond(request_id, {"sessionId": session_id})
        elif method == "session/load":
            session_id = params.get("sessionId", "sess-1")
            update(
                session_id,
                {
                    "sessionUpdate": "user_message_chunk",
                    "content": {"type": "text", "text": "loaded user message"},
                },
            )
            respond(request_id, {})
        elif method == "session/set_mode":
            respond(request_id, {})
        elif method == "session/set_config_option":
            respond(
                request_id,
                {
                    "configOptions": [
                        {
                            "id": params["configId"],
                            "name": "Model",
                            "type": "select",
                            "currentValue": params["value"],
                            "options": [],
                        }
                    ]
                },
            )
        elif method == "session/prompt":
            session_id = params.get("sessionId", session_id)
            text = (params.get("prompt") or [{}])[0].get("text", "")
            if text == "die":
                sys.exit(3)
            pending_prompt = request_id
            update(
                session_id,
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Hello "},
                },
            )
            update(
                session_id,
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc-1",
                    "title": "fake-tool",
                    "status": "pending",
                },
            )
            if text != "wait":
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": "perm-1",
                        "method": "session/request_permission",
                        "params": {
                            "sessionId": session_id,
                            "toolCall": {
                                "toolCallId": "tc-1",
                                "title": "fake-tool",
                                "status": "in_progress",
                            },
                            "options": [
                                {
                                    "optionId": "allow-1",
                                    "name": "Allow once",
                                    "kind": "allow_once",
                                }
                            ],
                        },
                    }
                )
                # Two more chunks stream while the permission request is still
                # outstanding, then a second agent-to-client request lands.
                for piece in ("a", "b"):
                    update(
                        session_id,
                        {
                            "sessionUpdate": "agent_message_chunk",
                            "content": {"type": "text", "text": piece},
                        },
                    )
                send(
                    {
                        "jsonrpc": "2.0",
                        "id": "fs-1",
                        "method": "fs/read_text_file",
                        "params": {"sessionId": session_id, "path": "/etc/hostname"},
                    }
                )
        else:
            send(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "Method not found"},
                }
            )
    elif "method" in msg:
        if msg["method"] == "session/cancel" and pending_prompt is not None:
            respond(pending_prompt, {"stopReason": "cancelled"})
            pending_prompt = None
    elif "id" in msg:
        if msg.get("id") == "perm-1":
            answered["perm"] = True
        elif msg.get("id") == "fs-1":
            answered["fs"] = True
        # Finish the prompt once both agent-initiated requests were answered.
        if pending_prompt is not None and answered["perm"] and answered["fs"]:
            update(
                session_id,
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "tc-1",
                    "status": "completed",
                },
            )
            update(
                session_id,
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "world"},
                },
            )
            respond(pending_prompt, {"stopReason": "end_turn"})
            pending_prompt = None
