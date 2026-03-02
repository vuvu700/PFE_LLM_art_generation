import sys
import pathlib
base_dir = pathlib.Path(__file__).parent.parent.parent.as_posix()
sys.path.append(base_dir)
print(f"added {base_dir!r} to import paths")
