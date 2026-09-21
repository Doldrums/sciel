"""Public task-owned candidate contract, separate from optional run instructions."""
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
