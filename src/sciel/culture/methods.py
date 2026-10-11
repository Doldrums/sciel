"""Inherited research methods re-enter as unpublished tools of their author island and must be released again."""

from __future__ import annotations


def export_methods(engine) -> list:
    tools = engine.research_tools
    if tools is None:
        return []
    culture = tools.culture
    published = list(tools._state["published"])
    credit = culture.credit(published)
    methods = []
    for ident in published:
        if culture.retired(ident) or credit[ident] < 0:
            continue
        bundle = tools.library.get(ident)
        manifest = bundle["manifest"]
        protocol = manifest["protocol"]
        methods.append({"id": ident, "name": manifest["name"], "description": protocol["description"],
                        "code": bundle["files"]["tool.py"], "author_island": protocol["author_island"],
                        "parents": manifest["parents"], "dependencies": manifest["dependencies"],
                        "replay_policy": protocol["replay_policy"], "credit": credit[ident]})
    return methods


def import_methods(tools, methods: list, islands: int) -> list:
    """Parents are lineage metadata: they follow re-registration and parents left behind by the donor are dropped.
    Code imports dependencies by ID, so their new IDs are written into the code; a method whose dependency was not
    passed on cannot run and is not imported."""
    known, renamed, imported = {m["id"] for m in methods}, {}, []
    pending = list(methods)
    while pending:
        ready = [m for m in pending if all(d in renamed or d not in known for d in m["parents"] + m["dependencies"])]
        if not ready:
            break
        for m in ready:
            pending.remove(m)
            if any(d not in renamed for d in m["dependencies"]):
                continue
            code = m["code"]
            for d in m["dependencies"]:
                code = code.replace(d, renamed[d])
            bundle = tools.library.register(m["name"], m["description"], code, m["author_island"] % islands,
                                            [renamed[p] for p in m["parents"] if p in renamed],
                                            dependencies=[renamed[d] for d in m["dependencies"]],
                                            replay_policy=m["replay_policy"])
            renamed[m["id"]] = bundle["id"]
            imported.append(bundle["id"])
            if bundle["id"] not in tools._state["registered"]:
                tools._state["registered"].append(bundle["id"])
    return imported
