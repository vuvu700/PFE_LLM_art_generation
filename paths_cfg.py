import os, os.path, sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(".env")


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
TOKENIZER_SAVE_DIRECTORY = joinAndEnsure(
    CURRENT_DIRECTORY, os.getenv('TOKENIZER_SAVE_DIRECTORY', default="tokenizer_save"))
