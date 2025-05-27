# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# WARNING: while you can read this file, however editing this file directly
# will stop your generation and you will stop abruptly and fail!
#
# If you want to add a new stop token for the next agent itertion, then you
# should append it to this file using a terminal tool like:
# echo 'NEW_STOP_TOKEN = "</NEW_STOP>"' >> tools/stop_tokens.py
#
# If you want to remove one, then make a line edit using something like:
# sed -i '<line_number>d' tools/stop_tokens.py.
# Note that the first token, TOOL_STOP_TOKEN, is on line 14 of this file after
# this comment is counted. To delete it, you'd do:
# sed -i '14d' tools/stop_tokens.py.

TOOL_STOP_TOKEN = "</TOOL_CALL>"
AGENT_STOP_TOKEN = "</AGENT_CALL>"
OVERSEER_STOP_TOKEN = "</OVERSEER_JUDGEMENT>"
