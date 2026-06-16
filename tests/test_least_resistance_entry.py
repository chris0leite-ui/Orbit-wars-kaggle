"""Guard: least_resistance's Kaggle entry point must be `agent`.

kaggle_environments loads an agent file by selecting the LAST top-level
callable in the module (kaggle_environments/agent.py:get_last_callable ->
`[v for v in env.values() if callable(v)][-1]`). If any function or class is
defined below `agent`, that helper becomes the entry point and the agent
returns a non-move value every turn -> idles silently (validates COMPLETE in
self-play, but scores ~floor on the real ladder). This regression shipped once
(sub 53740037, scored 332); this test makes it loud and fast.
"""
import ast
import os

_MAIN = os.path.join(os.path.dirname(__file__), "..", "agents",
                     "least_resistance", "main.py")


def test_agent_is_last_top_level_callable():
    tree = ast.parse(open(_MAIN).read())
    callables = [n.name for n in tree.body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                   ast.ClassDef))]
    assert callables, "no top-level callables found in main.py"
    assert callables[-1] == "agent", (
        "kaggle_environments uses the LAST top-level callable as the entry "
        f"point; it must be `agent` but is `{callables[-1]}`. Move every "
        "module-level def/class above agent() (see the header comment there)."
    )
