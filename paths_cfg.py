import os, os.path, sys
from pathlib import Path

def joinAndEnsure(src:"str|Path", *other:"str|Path")->Path:
    new = (src if isinstance(src, Path) else Path(src))
    for nextDir in other:
        new = new.joinpath(nextDir)
        if new.exists() is False:
            os.mkdir(new)
    assert new.exists(), f"the path: {new.as_posix()!r} don't exist (couldn't be crated)"
    return new


CURRENT_DIRECTORY = Path(__file__).parent

### logs
LOGS_DIRECTORY = joinAndEnsure(CURRENT_DIRECTORY, "logs")
