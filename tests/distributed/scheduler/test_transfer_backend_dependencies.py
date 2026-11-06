"""Backend imports must not load optional dependencies for other backends."""
import subprocess
import sys
import textwrap

import pytest


def _fresh_process(body):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_mooncake_protocol_does_not_import_transfer_queue():
    _fresh_process('''
        import importlib.abc
        import sys
        class BlockTransferQueue(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'transfer_queue' or fullname.startswith('transfer_queue.'):
                    raise ModuleNotFoundError('transfer_queue is absent', name='transfer_queue')
        sys.meta_path.insert(0, BlockTransferQueue())
        from roll.distributed.scheduler import transfer_backend
        from roll.distributed.scheduler.protocol import DataProto
        assert transfer_backend.tq is None
        try:
            transfer_backend.TransferQueueClient()
        except ImportError as exc:
            expected = 'Python 3.13' if sys.version_info >= (3, 13) else 'requires the transfer_queue package'
            assert expected in str(exc), exc
        else:
            raise AssertionError('missing optional package was accepted')
    ''')


def test_transfer_queue_is_loaded_on_first_use_and_cached(monkeypatch):
    import types
    from roll.distributed.scheduler import transfer_backend

    calls = []
    optional = types.ModuleType('transfer_queue')
    optional.init = lambda *args: calls.append(args)
    monkeypatch.setattr(transfer_backend, 'tq', None)
    monkeypatch.setattr(transfer_backend, 'sys', types.SimpleNamespace(version_info=(3, 12)))
    monkeypatch.setitem(sys.modules, 'transfer_queue', optional)

    transfer_backend.TransferQueueClient()
    assert transfer_backend.tq is optional
    monkeypatch.delitem(sys.modules, 'transfer_queue')
    transfer_backend.TransferQueueClient()
    assert calls == [(), ()]


def test_transfer_queue_server_checks_dependency_before_initializing(monkeypatch):
    from roll.distributed.scheduler import transfer_backend

    def unavailable():
        raise ImportError('optional dependency unavailable')

    monkeypatch.setattr(transfer_backend, '_check_transfer_queue_available', unavailable)
    with pytest.raises(ImportError, match='optional dependency unavailable'):
        transfer_backend.init_transfer_queue_server({})


def test_transfer_queue_keeps_internal_import_error(monkeypatch):
    import builtins
    import types
    from roll.distributed.scheduler import transfer_backend

    original_import = builtins.__import__
    def import_with_missing_dependency(name, *args, **kwargs):
        if name == 'transfer_queue':
            raise ModuleNotFoundError('missing internal dependency', name='dependency_inside_transfer_queue')
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(transfer_backend, 'tq', None)
    monkeypatch.setattr(transfer_backend, 'sys', types.SimpleNamespace(version_info=(3, 12)))
    monkeypatch.setattr(builtins, '__import__', import_with_missing_dependency)
    with pytest.raises(ModuleNotFoundError) as exc:
        transfer_backend._check_transfer_queue_available()
    assert exc.value.name == 'dependency_inside_transfer_queue'


def test_transfer_queue_keeps_python_version_restriction(monkeypatch):
    import types
    from roll.distributed.scheduler import transfer_backend

    monkeypatch.setattr(transfer_backend, 'sys', types.SimpleNamespace(version_info=(3, 13)))
    with pytest.raises(ImportError, match='Python 3.13'):
        transfer_backend._check_transfer_queue_available()
