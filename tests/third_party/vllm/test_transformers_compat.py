"""The Qwen3.5 backport patch is only needed with Transformers4."""
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize('vllm_version,transformers_version,expect_patch', [
    ('0.16.0', '4.57.1', True),
    ('0.32.0', '5.20.2', False),
    ('0.15.0', '4.57.1', False),
])
def test_qwen_backport_patch_version_boundary(vllm_version, transformers_version, expect_patch):
    pytest.importorskip('vllm')
    script = textwrap.dedent('''
        import importlib.abc
        import sys
        import transformers
        import vllm
        vllm.__version__, transformers.__version__ = sys.argv[1:3]
        class PatchSelected(Exception):
            pass
        class ObservePatch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'roll.third_party.vllm.patch_transformers':
                    raise PatchSelected()
        sys.meta_path.insert(0, ObservePatch())
        selected = False
        try:
            import roll.third_party.vllm
        except PatchSelected:
            selected = True
        assert selected == (sys.argv[3] == 'True'), selected
    ''')
    result = subprocess.run(
        [sys.executable, '-c', script, vllm_version, transformers_version, str(expect_patch)],
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
