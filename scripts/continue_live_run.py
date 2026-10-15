"""Start or resume a live experiment; back off after transient provider loss (see sciel.cli.supervise)."""

from sciel.cli.supervise import main

if __name__ == '__main__':
    raise SystemExit(main())
