"""Start or resume a live experiment; back off after transient provider loss (see sciel.cli.supervise)."""

import getpass  # noqa: F401  re-exported for callers that patch the prompt
import os  # noqa: F401

from sciel.cli.supervise import RuntimeStatus, backoff, ensure_credentials, main  # noqa: F401

if __name__ == '__main__':
    raise SystemExit(main())
