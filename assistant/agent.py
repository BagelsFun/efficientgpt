"""Agent loop: retrieval-augmented prompt, tool calls under permission control, bounded steps, full trace.

The trace records which answers used tools or retrieval, so benchmarks can
report model-only and tool-assisted results separately.
"""

import json
from dataclasses import dataclass, field

from .tools import ToolError

DEFAULT_SYSTEM = (
    'You are EfficientGPT Assistant, a careful engineering assistant running locally. '
    'Be concise and correct. Use tools when they help: read files instead of guessing, and say so when you are unsure. '
    'Never claim to have run code or tests unless a tool result shows it.')
# Measured on assistant-agent-v1.2 (EXPERIMENT_LOG ASSIST-05): cuts model-only unsupported answers of Qwen3-1.7B nf4 from
# 11/20 to 4/20 (McNemar p = 0.016) and qwen3.5:4b from 3/20 to 0/20, with tool-assisted accuracy unchanged (20/20; 17 -> 16).
CALIBRATION_RULE = (
    ' Calibration rule: state a specific value, name or number only if it appears in the provided context or a tool result. '
    'If it does not, do not guess or give a typical value: say you do not know and name the file to read.')
# v2 (ASSIST-06): with v1, Qwen3-1.7B answered tool-enabled coding tasks with "I don't have access to the files" and no
# tool call (assistant-coding-v1, 6/10). When tools are offered, the rule must point at them instead of at the user.
TOOL_RULE = (' When tools are available, you do have access to the workspace: use read_file / search_code / run_unittests '
             'to look things up before answering, and never say you cannot access the files.')
INTERACTIVE_SYSTEM = DEFAULT_SYSTEM + CALIBRATION_RULE  # CLI default; benchmarks name their prompt explicitly


@dataclass
class Turn:
    answer: str
    steps: list = field(default_factory=list)   # per model call: stats + tool calls
    retrieved: list = field(default_factory=list)
    used_tools: bool = False


class Agent:
    def __init__(self, backend, context, toolbox=None, retriever=None, max_steps=6, rag_k=4,
                 use_rag=True, use_tools=True, think=False, max_tokens=768, temperature=0.0,
                 tool_rule=True, force_final=True):
        self.backend, self.context, self.toolbox, self.retriever = backend, context, toolbox, retriever
        self.max_steps, self.rag_k = max_steps, rag_k
        self.use_rag, self.use_tools, self.think = use_rag, use_tools, think
        self.max_tokens, self.temperature = max_tokens, temperature
        # tool_rule: add TOOL_RULE on turns that offer tools. force_final: when the tool-step limit is hit, make one
        # last call without tools asking for the answer (qwen3.5 otherwise looped on read_file; ASSIST-06).
        self.tool_rule, self.force_final = tool_rule, force_final

    def ask(self, user_text, on_token=None):
        turn = Turn('')
        extra = ''
        if self.use_rag and self.retriever is not None:
            hits = self.retriever.search(user_text, self.rag_k)
            turn.retrieved = [f'{c.path}:{c.start_line}-{c.end_line}' for c in hits]
            extra = 'Possibly relevant workspace excerpts (may be incomplete):\n' + self.retriever.format(hits, 5000)
        self.context.add({'role': 'user', 'content': user_text})
        tools = self.toolbox.schemas() if (self.use_tools and self.toolbox) else None
        base_extra = extra
        if tools and self.tool_rule:
            extra = (extra + '\n\n' if extra else '') + TOOL_RULE.strip()
        for _ in range(self.max_steps):
            reply = self.backend.chat(self.context.build(extra), tools=tools, max_tokens=self.max_tokens,
                                      temperature=self.temperature, think=self.think, on_token=on_token)
            turn.steps.append({'stats': reply.stats.__dict__, 'tool_calls': reply.tool_calls})
            if not reply.tool_calls:
                self.context.add({'role': 'assistant', 'content': reply.text})
                turn.answer = reply.text
                return turn
            turn.used_tools = True
            self.context.add({'role': 'assistant', 'content': reply.text,
                              'tool_calls': [{'function': {'name': c['name'], 'arguments': c['arguments']}} for c in reply.tool_calls]})
            for c in reply.tool_calls:
                if not self.toolbox:
                    result = 'ERROR: tools disabled'
                else:
                    try:
                        result = self.toolbox.call(c['name'], c['arguments'])
                    except ToolError as e:  # unknown tool or malformed call: tell the model, don't crash
                        result = f'ERROR: {e}'
                turn.steps[-1].setdefault('tool_results', []).append(str(result)[:300])  # for evaluation traces
                self.context.add({'role': 'tool', 'content': str(result)[:6000], 'tool_name': c['name']})
        if self.force_final:
            self.context.add({'role': 'user', 'content': 'Tool-step limit reached. Do not call tools. Give your final '
                                                         'answer now, based on the tool results above.'})
            # no TOOL_RULE here: it tells the model to use tools, contradicting this call (review V1 #2)
            reply = self.backend.chat(self.context.build(base_extra), tools=None, max_tokens=self.max_tokens,
                                      temperature=self.temperature, think=self.think, on_token=on_token)
            # tool calls in a forced reply are never executed: record them apart so they are not counted as tool use
            turn.steps.append({'stats': reply.stats.__dict__, 'tool_calls': [], 'forced_final': True,
                               'ignored_tool_calls': reply.tool_calls})
            turn.answer = reply.text or '(stopped: tool-step limit reached)'
        else:
            turn.answer = '(stopped: tool-step limit reached)'
        self.context.add({'role': 'assistant', 'content': turn.answer})
        return turn


def trace_json(turn):
    return json.dumps({'answer': turn.answer, 'retrieved': turn.retrieved, 'used_tools': turn.used_tools,
                       'steps': turn.steps}, default=str)
