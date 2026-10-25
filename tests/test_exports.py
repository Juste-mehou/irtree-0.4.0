"""Top-level exports; load_data is loaded lazily (pandas only when used)."""
import subprocess
import sys

import irtree


def test_node_factories_are_exported():
    assert irtree.make_node("logit-2pl").__class__ is irtree.LogitTwoPLNode
    assert irtree.make_node("logit-3pl").__class__ is irtree.LogitThreePLNode
    assert "pcm" in irtree.NODE_TYPES
    assert set(irtree.__all__) <= set(dir(irtree)) | {"load_data"}


def test_load_data_is_lazy():
    code = ("import sys, irtree; assert 'pandas' not in sys.modules; "
            "from irtree import load_data; assert load_data.__module__ == 'irtree.ingest'")
    subprocess.run([sys.executable, "-c", code], check=True)
