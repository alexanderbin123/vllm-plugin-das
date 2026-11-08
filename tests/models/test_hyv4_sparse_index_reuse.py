"""CPU tests for the lifetime of a HYV4 integer index mapping.

Load the policy alone so these tests need neither a GPU nor platform imports.
"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest


MODULE = Path(__file__).parents[2] / "vllm_hcu/models/hy_v4/sparse_index_reuse.py"
spec = importlib.util.spec_from_file_location("hyv4_sparse_index_reuse", MODULE)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


class IndexView:
    def __init__(self, pointer=1024, shape=(32, 2048), strides=(2048, 1)):
        self.pointer, self.shape, self.strides = pointer, shape, strides

    def data_ptr(self):
        return self.pointer

    def stride(self):
        return self.strides


class MappingLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.context, self.metadata = SimpleNamespace(), SimpleNamespace()
        self.indices = IndexView()
        self.computed = []

    def convert(self, *, context=None, metadata=None, indices=None,
                producer=False, allowed=True):
        def compute():
            result = (object(), object())
            self.computed.append(result)
            return result
        return policy.reuse_or_compute(
            self.context if context is None else context,
            self.metadata if metadata is None else metadata,
            self.indices if indices is None else indices,
            is_producer=producer, allow_reuse=allowed, compute=compute,
        )

    def test_consumers_reuse_the_current_producer_mapping(self):
        produced = self.convert(producer=True)
        self.assertIs(self.convert(), produced)
        self.assertIs(self.convert(), produced)
        self.assertEqual(len(self.computed), 1)

    def test_each_producer_refreshes_even_with_the_same_buffer_and_metadata(self):
        old = self.convert(producer=True)
        fresh = self.convert(producer=True)
        self.assertIsNot(fresh, old)
        self.assertIs(self.convert(), fresh)
        self.assertEqual(len(self.computed), 2)

    def test_new_forward_cannot_reuse_previous_forward_output(self):
        old = self.convert(producer=True)
        fresh = self.convert(context=SimpleNamespace())
        self.assertIsNot(fresh, old)

    def test_equal_looking_metadata_is_not_the_same_mapping(self):
        old = self.convert(producer=True)
        fresh = self.convert(metadata=SimpleNamespace())
        self.assertIsNot(fresh, old)

    def test_pointer_shape_and_stride_changes_each_require_conversion(self):
        views = [IndexView(pointer=2048), IndexView(shape=(16, 2048)),
                 IndexView(strides=(4096, 1))]
        for view in views:
            old = self.convert(producer=True)
            self.assertIsNot(self.convert(indices=view), old)

    def test_capture_or_full_mode_invalidates_any_earlier_eager_mapping(self):
        eager = self.convert(producer=True)
        captured = self.convert(producer=True, allowed=False)
        self.assertIsNot(captured, eager)
        after = self.convert()
        self.assertIsNot(after, eager)
        self.assertIsNot(after, captured)
        self.assertIs(self.convert(), after)

    def test_disabled_consumer_does_not_read_or_populate_cache(self):
        first = self.convert(allowed=False)
        self.assertIsNot(self.convert(allowed=False), first)
        self.assertEqual(len(self.computed), 2)

    def test_missing_context_does_not_reuse(self):
        outputs = []
        for _ in range(2):
            outputs.append(policy.reuse_or_compute(
                None, self.metadata, self.indices, is_producer=False,
                allow_reuse=True, compute=lambda: object(),
            ))
        self.assertIsNot(outputs[0], outputs[1])

    def test_failed_producer_invalidates_the_old_eager_entry(self):
        old = self.convert(producer=True)
        def fail():
            raise RuntimeError("conversion failed")
        with self.assertRaisesRegex(RuntimeError, "conversion failed"):
            policy.reuse_or_compute(
                self.context, self.metadata, self.indices, is_producer=True,
                allow_reuse=True, compute=fail,
            )
        self.assertIsNot(self.convert(), old)

    def test_another_metadata_entry_cannot_survive_a_producer_refresh(self):
        self.convert(producer=True)
        other_metadata = SimpleNamespace()
        old_other = self.convert(metadata=other_metadata)
        self.convert(producer=True)
        self.assertIsNot(self.convert(metadata=other_metadata), old_other)
        self.assertEqual(len(self.computed), 4)


if __name__ == "__main__":
    unittest.main()
