"""Trusted loader source shared by Docker and source materialization.

Importing this module does not import any method. The factory is self-contained
so its exact source may be embedded in an immutable composed candidate.
"""


def make_method_loader(methods):
    import builtins
    import importlib
    import importlib.abc
    import importlib.machinery
    import importlib.util
    from pathlib import Path
    import sys
    import types

    definitions = {ident: dict(spec) for ident, spec in methods.items()}
    functions = {}

    class BundleLoader(importlib.machinery.SourceFileLoader):
        def __init__(self, fullname, path, namespace, root):
            super().__init__(fullname, path)
            self.namespace, self.root = namespace, root

        def exec_module(self, module):
            original = builtins.__import__
            namespace, root = self.namespace, self.root

            def pinned_import(name, globals=None, locals=None, fromlist=(), level=0):
                first = name.split('.')[0]
                if level == 0 and ((root / (first + '.py')).is_file() or (root / first).is_dir()):
                    imported = original(namespace + '.' + name, globals, locals, fromlist or ('*',), 0)
                    return imported if fromlist else sys.modules[namespace + '.' + first]
                return original(name, globals, locals, fromlist, level)

            module.__dict__['__builtins__'] = {**vars(builtins), '__import__': pinned_import}
            super().exec_module(module)

    class BundleFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            first, _, rest = fullname.partition('.')
            if not first.startswith('_sciel_bundle_'):
                return None
            method_id = first[len('_sciel_bundle_'):]
            if method_id not in definitions:
                return None
            root = Path(definitions[method_id]['root'])
            location = root.joinpath(*rest.split('.')) if rest else root
            if location.is_dir():
                source = location / '__init__.py'
                if not source.is_file():
                    spec = importlib.machinery.ModuleSpec(fullname, loader=None, is_package=True)
                    spec.submodule_search_locations = [str(location)]
                    return spec
                package_paths = [str(location)]
            else:
                source = location.with_suffix('.py')
                package_paths = None
            if not source.is_file():
                return None
            loader = BundleLoader(fullname, str(source), first, root)
            return importlib.util.spec_from_file_location(fullname, source, loader=loader,
                                                          submodule_search_locations=package_paths)

    def invoke(method_id, value):
        if method_id not in definitions:
            raise ValueError('method is outside the pinned dependency closure')
        if method_id not in functions:
            filename, function = definitions[method_id]['entrypoint'].split(':')
            parts = filename[:-3].split('/')
            namespace = '_sciel_bundle_' + method_id
            root = Path(definitions[method_id]['root'])
            package = '.'.join([namespace, *parts[:-1]])
            if parts[-1] == '__init__':
                name, package_paths = package, [str((root / filename).parent)]
            else:
                importlib.import_module(package)
                name, package_paths = package + '.' + parts[-1], None
            # Entrypoint is an exact manifest file path, not a module-name
            # search: tool.v1.py must never resolve to another tool/v1.py.
            loader = BundleLoader(name, str(root / filename), namespace, root)
            spec = importlib.util.spec_from_file_location(name, root / filename, loader=loader,
                                                          submodule_search_locations=package_paths)
            module = importlib.util.module_from_spec(spec)
            module.__package__ = package
            sys.modules[name] = module
            loader.exec_module(module)
            functions[method_id] = getattr(module, function)
        return functions[method_id](value)

    sys.meta_path.insert(0, BundleFinder())
    for ident in definitions:
        name = 'm_' + ident
        module = types.ModuleType(name)
        module.METHOD_ID = ident
        module.invoke = lambda value, method_id=ident: invoke(method_id, value)
        sys.modules[name] = module
    return invoke
