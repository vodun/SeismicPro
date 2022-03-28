"""General indexer assertions"""

from seismicpro.src.utils.indexer import TraceIndexer, GatherIndexer


def assert_indexers_equal(left, right):
    """Check if two indexers are equal."""
    # Check if both left and right are of the same type
    assert type(left) is type(right)

    # Check types of passed indexers
    assert isinstance(left, (TraceIndexer, GatherIndexer))

    # Compare indexer attributes
    assert left.index.equals(right.index)
    assert left.unique_indices.equals(right.unique_indices)
    if isinstance(left, GatherIndexer):
        assert left.index_to_headers_pos == left.index_to_headers_pos
