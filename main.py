"""gta6-twitch-ccu-forecast-01: weighted comparable launch curves + pinball-aware bands."""
import json
import math
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Soft preference for big open-world / Rockstar-like launches when present.
PREFER = (
    "red dead",
    "gta",
    "grand theft",
    "cyberpunk",
    "elden",
    "witcher",
    "horizon",
    "spider-man",
    "spiderman",
    "god of war",
    "starfield",
    "assassin",
    "ghost of",
    "hogwarts",
)


def round2(x: float):
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def day_hours(comp: dict):
    ds = parse_ts(comp["day_start"])
    by = {}
    for h in comp.get("hourly") or []:
        delta = int((parse_ts(h["from"]) - ds).total_seconds() // 3600)
        if 0 <= delta < 24:
            by[delta] = max(1.0, float(h["viewers"]))
    if 0 not in by:
        return None
    out = []
    for i in range(24):
        if i not in by:
            return None
        out.append(by[i])
    return out


def weight_for(comp: dict, p1: float) -> float:
    name = (comp.get("name") or "").lower()
    w = 1.0
    for needle in PREFER:
        if needle in name:
            w += 1.25
            break
    hrs = day_hours(comp)
    if hrs:
        # Prefer launches whose hour-1 scale is within ~10x of ours.
        ratio = max(hrs[0], 1.0) / max(p1, 1.0)
        if ratio < 1:
            ratio = 1.0 / ratio
        w *= 1.0 / (1.0 + math.log(ratio))
    return w


def weighted_quantile(values, weights, q: float) -> float:
    pairs = sorted(zip(values, weights), key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    if total <= 0:
        return statistics.median(values)
    target = q * total
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= target:
            return v
    return pairs[-1][0]


def log_shape_rows(case: dict, p1: float):
    rows = []
    for c in case.get("comparables") or []:
        hrs = day_hours(c)
        if not hrs:
            continue
        shape = [math.log(hrs[i] / hrs[0]) for i in range(1, 24)]
        rows.append((shape, weight_for(c, p1)))
    return rows


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} levels mismatch")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} missing hour 1")
    p1 = max(1.0, float(first["viewers"]))
    rows = log_shape_rows(case, p1)

    # Mild diurnal prior if comps are thin: evening bump in UTC evening hours.
    hours_out = []
    for row in case["forecast"]:
        h = int(row["hour"])
        idx = h - 2
        # Persistence center in log space, plus comparable median path.
        persist = 0.0
        if rows and 0 <= idx < 23:
            vals = [s[idx] for s, _ in rows]
            ws = [w for _, w in rows]
            center = weighted_quantile(vals, ws, 0.5)
            q05 = weighted_quantile(vals, ws, 0.05)
            q25 = weighted_quantile(vals, ws, 0.25)
            q75 = weighted_quantile(vals, ws, 0.75)
            q95 = weighted_quantile(vals, ws, 0.95)
            # Shrink extreme tails toward center (pinball punishes over-wide bands).
            shrink = 0.65
            q05 = center + (q05 - center) * shrink
            q25 = center + (q25 - center) * shrink
            q75 = center + (q75 - center) * shrink
            q95 = center + (q95 - center) * shrink
            # Blend 85% comps / 15% flat persistence.
            blend = 0.85
            center = blend * center + (1 - blend) * persist
            q05 = blend * q05 + (1 - blend) * (persist - 0.15 * math.sqrt(h - 1) * 1.645)
            q25 = blend * q25 + (1 - blend) * (persist - 0.15 * math.sqrt(h - 1) * 0.674)
            q75 = blend * q75 + (1 - blend) * (persist + 0.15 * math.sqrt(h - 1) * 0.674)
            q95 = blend * q95 + (1 - blend) * (persist + 0.15 * math.sqrt(h - 1) * 1.645)
            # Floor minimum width so 90% band is not absurdly tight.
            min_half = 0.06 * math.sqrt(h - 1)
            if (q95 - q05) < 2 * min_half:
                mid = 0.5 * (q05 + q95)
                q05, q95 = mid - min_half, mid + min_half
            if (q75 - q25) < min_half:
                mid = 0.5 * (q25 + q75)
                q25, q75 = mid - 0.5 * min_half, mid + 0.5 * min_half
            logs = [q05, q25, center, q75, q95]
        else:
            logs = [z * 0.15 * math.sqrt(h - 1) for z in Z]

        qs = [round2(max(1.0, min(1e9, p1 * math.exp(v)))) for v in logs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours_out.append({"hour": h, "q": qs})
    return {"hours": hours_out}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
