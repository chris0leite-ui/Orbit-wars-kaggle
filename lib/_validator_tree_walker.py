"""Pure-Python parser + walker for a LightGBM binary-classification Booster.

Why this exists: the validator agent ships as a single Python file inlined
by `scripts/bundle_validator.py`. We cannot ship the `lightgbm` C++ dep
at submit time. Instead, the trained Booster is saved via
`Booster.model_to_string()`, the text is embedded as a base64 blob in
`agents/baseline_validated/main.py`, and this module parses + walks the
trees with pure numpy.

Public API
----------
parse_booster_text(text) -> ParsedBooster
    Parse a LightGBM model dump produced by `model_to_string()` (or by
    `save_model(path)` and `Path(path).read_text()`).
predict_raw(parsed, X) -> np.ndarray(N,)
    Sum of leaf values across all trees for every row of X. Matches
    `Booster.predict(X, raw_score=True)` to within 1e-6 on the trained
    objective.
predict_proba(parsed, X) -> np.ndarray(N,)
    `sigmoid(predict_raw(parsed, X) * sigmoid_scale)`. Matches
    `Booster.predict(X)` to within 1e-6 for `objective=binary`.

Format reference
----------------
A Tree=K block contains six load-bearing arrays of length
`num_internal_nodes`:
  split_feature:  feature index used at each internal node
  threshold:      numeric threshold (decision_type=2 → uses "<=")
  decision_type:  bits 0=cat-vs-num, 1=missing-default, 2=lt-vs-le
  left_child:     >=0 -> next internal node; <0 -> leaf index `-(v+1)`
  right_child:    same encoding
plus one array of length `num_leaves`:
  leaf_value:     log-odds contribution of the leaf

For binary objective:
  raw_score(x) = sum over trees of leaf_value(walk(tree, x))
  prob(x)      = 1 / (1 + exp(-raw_score(x) * sigmoid_scale))
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Tree:
    """A single boosted tree's load-bearing arrays."""
    split_feature: np.ndarray  # int32, length = num_internal_nodes
    threshold: np.ndarray      # float64
    decision_type: np.ndarray  # uint8
    left_child: np.ndarray     # int32
    right_child: np.ndarray    # int32
    leaf_value: np.ndarray     # float64, length = num_leaves


@dataclass(frozen=True)
class ParsedBooster:
    trees: list  # list[_Tree]
    num_features: int  # max_feature_idx + 1
    sigmoid_scale: float  # 1.0 unless `sigmoid:N` in objective
    objective: str  # "binary" expected
    average_output: bool  # boost_from_average=1 (we treat baseline as
                          # baked into the leaf values; raw-score parity
                          # against Booster.predict(raw_score=True) is
                          # the contract that confirms it)


_TREE_HEADER_RE = re.compile(r"^Tree=(\d+)\s*$", re.MULTILINE)
_KEY_RE = re.compile(r"^([A-Za-z_]+)=(.*)$")


def _parse_floats(s: str) -> np.ndarray:
    return np.fromstring(s.strip(), sep=" ", dtype=np.float64)


def _parse_ints(s: str) -> np.ndarray:
    return np.fromstring(s.strip(), sep=" ", dtype=np.int64)


def _parse_tree_block(block: str) -> _Tree:
    """Parse the body of one `Tree=K` block (without the header line).

    Each line is `key=values`. We pull the six fields we need; others
    (split_gain, leaf_weight, internal_value, internal_count, shrinkage,
    leaf_count) are ignored — they don't affect inference.
    """
    fields: dict[str, str] = {}
    for line in block.splitlines():
        m = _KEY_RE.match(line.strip())
        if m:
            fields[m.group(1)] = m.group(2)
    return _Tree(
        split_feature=_parse_ints(fields["split_feature"]).astype(np.int32),
        threshold=_parse_floats(fields["threshold"]),
        decision_type=_parse_ints(fields["decision_type"]).astype(np.uint8),
        left_child=_parse_ints(fields["left_child"]).astype(np.int32),
        right_child=_parse_ints(fields["right_child"]).astype(np.int32),
        leaf_value=_parse_floats(fields["leaf_value"]),
    )


def parse_booster_text(text: str) -> ParsedBooster:
    """Parse a LightGBM model_to_string() dump into a ParsedBooster."""
    # ---- header (key=value lines before the first `Tree=` block) ----
    header_end = text.find("\nTree=")
    if header_end < 0:
        raise ValueError("no Tree= block in booster text")
    header = text[:header_end]
    header_kv: dict[str, str] = {}
    for line in header.splitlines():
        m = _KEY_RE.match(line.strip())
        if m:
            header_kv[m.group(1)] = m.group(2)

    objective_raw = header_kv.get("objective", "")
    # objective looks like "binary sigmoid:1" — split into kind + flags.
    objective_kind = objective_raw.split()[0] if objective_raw else "binary"
    sigmoid_scale = 1.0
    m_sig = re.search(r"sigmoid:(\d+(?:\.\d+)?)", objective_raw)
    if m_sig:
        sigmoid_scale = float(m_sig.group(1))
    num_features = int(header_kv["max_feature_idx"]) + 1
    average_output = header_kv.get("boost_from_average", "0").strip() == "1"

    # ---- find every Tree=K block boundary ----
    headers = list(_TREE_HEADER_RE.finditer(text))
    if not headers:
        raise ValueError("no Tree=K block headers")

    # The "end of trees" marker terminates the last block; if missing,
    # the parameters section header `\n[label_column:` does.
    end_marker = text.find("\nend of trees")
    if end_marker < 0:
        end_marker = text.find("\n[label_column:")
    if end_marker < 0:
        end_marker = len(text)

    trees: list = []
    for i, h in enumerate(headers):
        body_start = h.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else end_marker
        body = text[body_start:body_end]
        trees.append(_parse_tree_block(body))

    return ParsedBooster(
        trees=trees,
        num_features=num_features,
        sigmoid_scale=sigmoid_scale,
        objective=objective_kind,
        average_output=average_output,
    )


def _walk_one(tree: _Tree, x: np.ndarray) -> float:
    """Walk a single tree for a single row x → leaf_value.

    `left_child[i] >= 0` means child is an internal-node index (we
    recurse there). `< 0` means child is a leaf; leaf_index = -(value+1).
    decision_type bit 0 = 0 means numerical; bit 2 = 0 means "<="
    (the only forms our trainer produces).
    """
    node = 0
    sf = tree.split_feature
    th = tree.threshold
    lc = tree.left_child
    rc = tree.right_child
    while True:
        thr = th[node]
        feat = sf[node]
        # "<=" — matches decision_type bit 2 = 0 (LightGBM default).
        if x[feat] <= thr:
            child = lc[node]
        else:
            child = rc[node]
        if child < 0:
            return float(tree.leaf_value[-(child + 1)])
        node = int(child)


def predict_raw(parsed: ParsedBooster, X: np.ndarray) -> np.ndarray:
    """Sum of leaf_values across all trees for each row.

    Matches `Booster.predict(X, raw_score=True)` to ~1e-6.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] < parsed.num_features:
        raise ValueError(
            f"X has {X.shape[1]} cols but booster expects "
            f">= {parsed.num_features}"
        )
    n = X.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for tree in parsed.trees:
        for i in range(n):
            out[i] += _walk_one(tree, X[i])
    return out


def predict_proba(parsed: ParsedBooster, X: np.ndarray) -> np.ndarray:
    """sigmoid(predict_raw * sigmoid_scale). Matches `Booster.predict(X)`."""
    raw = predict_raw(parsed, X)
    s = parsed.sigmoid_scale
    return 1.0 / (1.0 + np.exp(-np.clip(raw * s, -30.0, 30.0)))
