#!/usr/bin/env python3
"""Mock MCP server for testing voxtty --mock-mcp flag.

Implements the MCP protocol over stdio (JSON-RPC 2.0).
Provides tools: get_time, echo, calculate, weather, random_fact, dice_roll.
"""

import json
import math
import random
import sys
from datetime import datetime


FACTS = [
    "Honey never spoils. Archaeologists have found 3000-year-old honey in Egyptian tombs that was still edible.",
    "Octopuses have three hearts and blue blood.",
    "A group of flamingos is called a 'flamboyance'.",
    "The shortest war in history lasted 38 minutes between Britain and Zanzibar in 1896.",
    "Bananas are berries, but strawberries are not.",
    "There are more possible chess games than atoms in the observable universe.",
    "Cambodia's Angkor Wat is the largest religious monument in the world.",
    "The human nose can detect over 1 trillion different scents.",
    "Light takes 8 minutes and 20 seconds to travel from the Sun to Earth.",
    "A day on Venus is longer than a year on Venus.",
]


def send_response(response):
    """Write a JSON-RPC response to stdout."""
    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()


def text_result(req_id, text):
    """Helper to send a text content result."""
    send_response({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": {
            "content": [{"type": "text", "text": text}]
        },
    })


def handle_request(request):
    """Handle a single JSON-RPC request."""
    method = request.get("method", "")
    req_id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "voxtty-mock",
                    "version": "1.0.0",
                },
            },
        })
    elif method == "notifications/initialized":
        pass  # Notification, no response needed
    elif method == "tools/list":
        send_response({
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "get_time",
                        "description": "Get the current date and time",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "format": {
                                    "type": "string",
                                    "description": "Output format: 'full' (date+time), 'date' (date only), 'time' (time only). Defaults to full.",
                                }
                            },
                        },
                    },
                    {
                        "name": "calculate",
                        "description": "Evaluate a math expression. Supports +, -, *, /, **, sqrt, sin, cos, tan, log, pi, e. Example: '2 * pi * 5' or 'sqrt(144)'",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "The math expression to evaluate",
                                }
                            },
                            "required": ["expression"],
                        },
                    },
                    {
                        "name": "weather",
                        "description": "Get the current weather for a city (mock data for testing)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "city": {
                                    "type": "string",
                                    "description": "City name, e.g., 'Phnom Penh', 'Tokyo', 'London'",
                                }
                            },
                            "required": ["city"],
                        },
                    },
                    {
                        "name": "random_fact",
                        "description": "Get a random fun fact",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                    {
                        "name": "dice_roll",
                        "description": "Roll dice. Supports standard notation like '2d6', '1d20', '3d8+5'",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "notation": {
                                    "type": "string",
                                    "description": "Dice notation, e.g., '2d6', '1d20+3'. Defaults to '1d6'.",
                                }
                            },
                        },
                    },
                    {
                        "name": "echo",
                        "description": "Echo back the provided message (useful for testing)",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "message": {
                                    "type": "string",
                                    "description": "The message to echo back",
                                }
                            },
                            "required": ["message"],
                        },
                    },
                ]
            },
        })
    elif method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        if tool_name == "get_time":
            fmt = arguments.get("format", "full")
            now = datetime.now()
            if fmt == "date":
                text_result(req_id, now.strftime("Today is %A, %B %d, %Y"))
            elif fmt == "time":
                text_result(req_id, now.strftime("Current time: %I:%M:%S %p"))
            else:
                text_result(req_id, now.strftime("It is %A, %B %d, %Y at %I:%M:%S %p"))

        elif tool_name == "calculate":
            expr = arguments.get("expression", "")
            try:
                safe_dict = {
                    "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
                    "tan": math.tan, "log": math.log, "log10": math.log10,
                    "abs": abs, "round": round, "pi": math.pi, "e": math.e,
                    "pow": pow, "ceil": math.ceil, "floor": math.floor,
                }
                result = eval(expr, {"__builtins__": {}}, safe_dict)  # noqa: S307
                text_result(req_id, f"{expr} = {result}")
            except Exception as e:
                text_result(req_id, f"Error evaluating '{expr}': {e}")

        elif tool_name == "weather":
            city = arguments.get("city", "Unknown")
            mock_weather = {
                "phnom penh": {"temp": 34, "condition": "Partly cloudy", "humidity": 72},
                "siem reap": {"temp": 33, "condition": "Sunny", "humidity": 65},
                "tokyo": {"temp": 22, "condition": "Clear", "humidity": 55},
                "london": {"temp": 15, "condition": "Overcast", "humidity": 80},
                "new york": {"temp": 25, "condition": "Sunny", "humidity": 60},
                "paris": {"temp": 18, "condition": "Light rain", "humidity": 75},
                "bangkok": {"temp": 35, "condition": "Hot and humid", "humidity": 78},
            }
            w = mock_weather.get(city.lower(), {
                "temp": random.randint(15, 35),
                "condition": random.choice(["Sunny", "Cloudy", "Rainy", "Windy"]),
                "humidity": random.randint(40, 90),
            })
            text_result(
                req_id,
                f"Weather in {city}: {w['temp']}°C, {w['condition']}, humidity {w['humidity']}%"
            )

        elif tool_name == "random_fact":
            text_result(req_id, random.choice(FACTS))

        elif tool_name == "dice_roll":
            notation = arguments.get("notation", "1d6")
            try:
                # Parse NdS+M notation
                bonus = 0
                if "+" in notation:
                    notation, bonus_str = notation.split("+", 1)
                    bonus = int(bonus_str)
                elif "-" in notation:
                    parts = notation.rsplit("-", 1)
                    notation = parts[0]
                    bonus = -int(parts[1])

                num, sides = notation.lower().split("d")
                num = int(num) if num else 1
                sides = int(sides)

                if num < 1 or num > 100 or sides < 2 or sides > 1000:
                    text_result(req_id, "Invalid dice: keep it between 1-100 dice with 2-1000 sides")
                else:
                    rolls = [random.randint(1, sides) for _ in range(num)]
                    total = sum(rolls) + bonus
                    bonus_str = f" + {bonus}" if bonus > 0 else f" - {abs(bonus)}" if bonus < 0 else ""
                    if num == 1:
                        text_result(req_id, f"Rolled 1d{sides}{bonus_str}: {total}")
                    else:
                        text_result(req_id, f"Rolled {num}d{sides}{bonus_str}: {rolls} = {total}")
            except (ValueError, IndexError):
                text_result(req_id, f"Invalid dice notation: '{notation}'. Use format like '2d6' or '1d20+3'")

        elif tool_name == "echo":
            msg = arguments.get("message", "")
            text_result(req_id, f"Echo: {msg}")

        else:
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Unknown tool: {tool_name}",
                },
            })
    else:
        if req_id is not None:
            send_response({
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            })


def main():
    """Main loop: read JSON-RPC requests from stdin, respond on stdout."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            handle_request(request)
        except json.JSONDecodeError:
            send_response({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": "Parse error"},
            })


if __name__ == "__main__":
    main()
