"""Conversation memory kept within a token budget (system prompt always kept, oldest turns dropped first)."""


def approx_tokens(text):
    """Fallback counter (~4 characters per token) when the backend has no tokenizer."""
    return max(1, len(text) // 4)


class ContextManager:
    def __init__(self, system, budget_tokens=6000, count=approx_tokens):
        self.system, self.budget, self.count = system, budget_tokens, count
        self.turns = []  # list of message dicts (user / assistant / tool)

    def add(self, message):
        self.turns.append(message)

    def reset(self):
        self.turns = []

    def _size(self, m):
        return self.count(m.get('content') or '') + 4

    def build(self, extra_system=''):
        """Messages for the next call; drops whole oldest turns until the budget fits."""
        system = self.system + (('\n\n' + extra_system) if extra_system else '')
        head = {'role': 'system', 'content': system}
        kept, used = [], self._size(head)
        for m in reversed(self.turns):
            s = self._size(m)
            if used + s > self.budget and kept:
                break
            kept.append(m)
            used += s
        kept.reverse()
        while kept and kept[0]['role'] == 'tool':  # never start with an orphaned tool result
            kept.pop(0)
        return [head] + kept

    def dropped(self):
        return len(self.turns) - (len(self.build()) - 1)
