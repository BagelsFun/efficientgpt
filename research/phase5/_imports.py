"""Keep imports of graph-specific implementations lazy for policy tests."""


def graph_classes():
    from ..phase4.decoder import GraphDecoder
    from ..phase4.compact_cache import CompactGraphDecoder
    return GraphDecoder, CompactGraphDecoder
