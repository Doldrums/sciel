"""Test boundaries isolate retained candidate imports without executing code."""

import sys
import types

import pytest


@pytest.fixture(scope='class')
def namespace_boundary():
    import sciel.candidate_modules as namespace

    old_name = namespace.__name__ + '._isolation_existing'
    new_name = namespace.__name__ + '._isolation_new'
    old = types.ModuleType(old_name)
    new = types.ModuleType(new_name)
    old_attr, new_attr = old_name.rsplit('.', 1)[1], new_name.rsplit('.', 1)[1]
    sys.modules[old_name] = old
    setattr(namespace, old_attr, old)
    try:
        yield namespace, new_name, new_attr, new
        # Class teardown follows the shared function-scoped fixture's teardown.
        assert new_name not in sys.modules
        assert not hasattr(namespace, new_attr)
        assert sys.modules[old_name] is old
        assert getattr(namespace, old_attr) is old
    finally:
        for name, attr in ((old_name, old_attr), (new_name, new_attr)):
            sys.modules.pop(name, None)
            if hasattr(namespace, attr):
                delattr(namespace, attr)


class TestCandidateNamespaceBoundary:
    def test_imports_remain_available_until_test_teardown(self, namespace_boundary):
        namespace, name, attr, module = namespace_boundary
        sys.modules[name] = module
        setattr(namespace, attr, module)
        assert sys.modules[name] is module
        assert getattr(namespace, attr) is module
