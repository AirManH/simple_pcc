from typing import Any, Iterable, Dict

import arithmeticcoding


class ContextModel:
    def __init__(self, contexts: Iterable, frequencies):
        self.freq_tables: Dict[Any, arithmeticcoding.SimpleFrequencyTable] = {}
        for context in contexts:
            sft = arithmeticcoding.SimpleFrequencyTable(frequencies[context])
            self.freq_tables[context] = sft

    def get(self, context, symbol=None):
        if symbol is None:
            return self.freq_tables[context]
        else:
            return self.freq_tables[context].get(symbol)

    def set(self, context, symbol, freq):
        self.freq_tables[context].set(symbol, freq)

    def increment(self, context, symbol):
        self.freq_tables[context].increment(symbol)
