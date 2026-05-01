import importlib.util
import os
import sys


def _run() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    src = os.path.abspath(os.path.join(here, "..", "..", "drugldm", "tanimoto_eval.py"))
    spec = importlib.util.spec_from_file_location("_tanimoto_eval_src", src)
    if spec is None or spec.loader is None:
        raise RuntimeError(src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tanimoto_eval_src"] = mod
    spec.loader.exec_module(mod)
    mod.main()


if __name__ == "__main__":
    _run()

