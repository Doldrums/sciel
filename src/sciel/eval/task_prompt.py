"""Public task-owned candidate contract, separate from optional run instructions."""

import os

MAX_TASK_PROMPT_CHARS = 1600
START = 'PUBLIC TASK CONTRACT (candidate interface, not research tool API):\n'
END = '\nEND PUBLIC TASK CONTRACT'
EXTRA = '\n\nADDITIONAL RUN INSTRUCTIONS:\n'


def validate_task_prompt(value):
    if value is None:
        return None
    if (type(value) is not str or not value.strip() or len(value) > MAX_TASK_PROMPT_CHARS
            or any(ord(c) < 32 and not c.isspace() for c in value)
            or START in value or END in value):
        raise ValueError('task_prompt must be nonempty public text of at most 1600 characters')
    value.encode('utf-8')
    return value


def compose_task_prompt(task, extra=None):
    public = validate_task_prompt(getattr(task, 'task_prompt', None))
    if public is None:
        return extra if extra is not None else task.qualified_name
    return START + public + END + (EXTRA + extra if extra else '')


def split_task_prompt(text):
    """Keep the complete bounded public block mandatory in compact planners.

    Historical snapshots without this block keep their original rendering. The
    complete composed string remains in the immutable tool input in both cases.
    """
    if not text.startswith(START):
        return '', text
    public, separator, extra = text[len(START):].partition(END)
    if not separator:
        raise ValueError('incomplete public task contract')
    validate_task_prompt(public)
    return START + public + END, extra.removeprefix(EXTRA)


def _first_sentence(doc, limit=160):
    paragraph = " ".join(doc.strip().split("\n\n")[0].split())
    end = paragraph.find(". ")
    sentence = paragraph[:end + 1] if end >= 0 else paragraph
    return sentence if len(sentence) <= limit else sentence[:limit - 3].rstrip() + "..."


def asset_summary(root):
    """The public names of a task's assets/*.py, read with ast without running them: per module its short constants
    and one line per function (signature and the first sentence of its docstring). Empty when there are none."""
    import ast
    folder = os.path.join(root, "assets")
    if not os.path.isdir(folder):
        return ""
    blocks = []
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".py") or name.startswith("_"):
            continue
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except (OSError, SyntaxError, ValueError):
            continue
        constants, functions = [], []
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.isupper()):
                value = ast.unparse(node.value)
                if len(value) <= 40:
                    constants.append(f"{node.targets[0].id} = {value}")
            elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
                doc = ast.get_docstring(node)
                functions.append(f"  {node.name}({ast.unparse(node.args)}){returns}"
                                 + (f": {_first_sentence(doc)}" if doc else ""))
        if functions or constants:
            head = name[:-3] + (": " + ", ".join(constants) if constants else "")
            blocks.append("\n".join([head] + functions))
    return "\n".join(blocks)
