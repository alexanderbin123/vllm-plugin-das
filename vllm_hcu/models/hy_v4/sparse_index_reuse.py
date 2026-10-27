# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward-local lifetime policy for converted sparse indices."""

_CONTEXT_ATTRIBUTE = "_hyv4_sparse_index_mapping"


def reuse_or_compute(context, metadata, indices, *, is_producer, allow_reuse, compute):
    """Reuse an integer mapping only in its current eager forward/producer group.

    The caller must disable reuse for capture/FULL graphs and any path where
    metadata can change within the forward. Producers always refresh, including
    repeated MTP calls that write the same raw buffer. Metadata is retained by
    reference; shape or pointer equality alone never establishes validity.
    """
    if context is None:
        return compute()
    if not allow_reuse:
        # A captured producer may follow an eager producer in the same context.
        # Its mapping must not leave an older eager cache entry usable later.
        setattr(context, _CONTEXT_ATTRIBUTE, None)
        return compute()

    key = (indices.data_ptr(), tuple(indices.shape), tuple(indices.stride()))
    entry = getattr(context, _CONTEXT_ATTRIBUTE, None)
    if not is_producer and entry is not None:
        prior_metadata, prior_key, result = entry
        if prior_metadata is metadata and prior_key == key:
            return result

    # Invalidate before dispatch, including a producer conversion that raises.
    setattr(context, _CONTEXT_ATTRIBUTE, None)
    result = compute()
    setattr(context, _CONTEXT_ATTRIBUTE, (metadata, key, result))
    return result
