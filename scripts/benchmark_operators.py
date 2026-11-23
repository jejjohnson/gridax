"""Forward / reverse-mode benchmark of the finitevolX operator surface.

For every operator it records, at one grid size:

* median wall time of the jitted forward, of the jitted VJP
  (``jax.vjp(op, *inputs)[1](cotangent)`` with a fixed random cotangent:
  one adjoint / reverse-mode application, forward included) and of the
  jitted JVP (``jax.jvp(op, inputs, tangents)``: one tangent-linear
  application, as in 4DVar's TLM),
* the number of XLA fusions and the temp-buffer bytes of each compiled
  program (``compiled.memory_analysis()``),
* the number of ``slice`` primitives in the forward jaxpr (one per stencil
  read; each becomes a ``pad`` in the transpose),
* an efficiency figure: the time an elementwise kernel moving the same
  number of bytes takes, divided by the operator's time (1.0 = at the
  memory roofline of this machine).

Usage::

    uv run python scripts/benchmark_operators.py --n 512 --out before.json
    uv run python scripts/benchmark_operators.py --n 512 --out after.json
    uv run python scripts/benchmark_operators.py --compare before.json after.json
    uv run python scripts/benchmark_operators.py --profile "advect.*weno5"

``--profile`` traces the VJP of the matching cases with ``jax.profiler`` and
prints the most expensive XLA kernels (fusions), which is how the hotspots in
``docs/design/research/stencil_performance_deep_dive.md`` were found.

It is a developer tool: nothing in the test suite imports it.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
import json
import re
import statistics
import sys
import time

import gridax as fvx
from gridax.schemes import TVD, WENO, Upwind
from gridax.solvers import masked_laplacian
import jax
import jax.numpy as jnp
import numpy as np


@dataclass
class Case:
    name: str
    fn: Callable
    n_in: int  # number of differentiable array inputs
    shape: tuple[int, ...]
    n_out: int = 1


def _cases(n: int, nz: int) -> list[Case]:
    g1 = fvx.CartesianCGrid1D.from_interior(n * n - 2, 1.0)
    g2 = fvx.CartesianCGrid2D.from_interior(n - 2, n - 2, 1.0, 1.0)
    g3 = fvx.CartesianCGrid3D.from_interior(n - 2, n - 2, nz - 2, 1.0, 1.0, 1.0)
    gs = fvx.SphericalCGrid2D.from_interior(
        n - 2, n - 2, lon_range=(0.0, 60.0), lat_range=(-30.0, 30.0)
    )
    s1, s2, s3 = (n * n,), (n, n), (nz, n, n)
    f2 = jnp.ones(s2)
    o1, o2, o3, sph = (fvx.GridOps(grid) for grid in (g1, g2, g3, gs))

    def case(label: str, ops, name: str, n_in: int, n_out: int = 1, **kwargs) -> Case:
        shape = {1: s1, 2: s2, 3: s3}[len(ops.grid.axes)]

        def fn(*arrays):
            return getattr(ops, name)(*arrays, **kwargs)

        return Case(f"{name}[{label}]", fn, n_in, shape, n_out)

    def move(label: str, ops, src: str, dst: str, axis: str | None = None) -> Case:
        kwargs = {"src": src, "dst": dst}
        if axis is None:
            return case(f"{label}, {src} to {dst}", ops, "interpolate", 1, **kwargs)
        return case(
            f"{label}, {axis}, {src} to {dst}",
            ops,
            "difference",
            1,
            axis=axis,
            **kwargs,
        )

    C = [move("1d", o1, "T", "U", "x"), case("1d", o1, "laplacian", 1)]
    C += [
        move("2d", o2, src, dst, axis)
        for axis, src, dst in (
            ("x", "T", "U"),
            ("y", "T", "V"),
            ("x", "U", "T"),
            ("y", "V", "T"),
        )
    ]
    C += [
        case("2d", o2, "divergence", 2),
        case("2d", o2, "curl", 2),
        case("2d", o2, "laplacian", 1),
        case("2d", o2, "grad_perp", 1, 2, at="T"),
        case("3d", o3, "divergence", 2),
        case("3d", o3, "laplacian", 1),
    ]
    C += [move("2d", o2, *pair) for pair in ("TU", "TX", "XT", "UV", "VU", "UT")]
    C += [
        Case("coriolis[2d]", lambda u, v, f: o2.coriolis(u, v, f=f), 3, s2, 2),
        case("3d", o3, "coriolis", 2, 2, f=f2),
    ]
    C += [
        case(f"2d, {form}", o2, "pv_flux", 3, 2, form=form)
        for form in ("energy", "enstrophy", "blend")
    ]
    C += [
        case("2d", o2, "kinetic_energy", 2),
        case("2d", o2, "okubo_weiss_parameter", 2),
        case("2d", o2, "arakawa_jacobian", 2),
        case("2d", o2, "diffuse", 1, kappa=1.0),
        Case("diffuse[2d, kappa field]", lambda h, k: o2.diffuse(h, kappa=k), 2, s2),
        case("2d", o2, "biharmonic", 1, kappa4=1.0),
        case("3d", o3, "diffuse", 1, kappa=1.0),
        case("2d, energy", o2, "advect_momentum", 2, 2, form="energy"),
        case("2d, enstrophy", o2, "advect_momentum", 2, 2, form="enstrophy"),
    ]
    schemes = {
        "upwind1": Upwind(1),
        "upwind3": Upwind(3),
        "weno3": WENO(3),
        "weno5": WENO(5),
        "wenoz5": WENO(5, "z"),
        "weno7": WENO(7),
        "van_leer": TVD("van_leer"),
    }
    C += [
        case(f"2d, {name}", o2, "advect", 3, scheme=scheme)
        for name, scheme in schemes.items()
    ]
    C.append(case("2d, weno5, open", o2, "advect", 3, scheme=WENO(5), wall="open"))
    # a basin with an island: exercises the adaptive-stencil (upwind_flux) path
    wet = np.ones(s2, dtype=bool)
    wet[n // 3 : n // 2, n // 3 : n // 2] = False
    masked = fvx.GridOps(g2, fvx.Mask2D.from_mask(wet))
    C += [
        case("2d, weno5, masked", masked, "advect", 3, scheme=WENO(5)),
        Case(
            "advect[1d, weno5]",
            lambda h, u: o1.advect(h, u, None, scheme=WENO(5)),
            2,
            s1,
        ),
        case("3d, weno5", o3, "advect", 3, scheme=WENO(5)),
    ]
    C += [
        move("sphere", sph, "T", "U", "x"),
        case("sphere", sph, "divergence", 2),
        case("sphere", sph, "laplacian", 1),
        case("sphere", sph, "curl", 2),
        case("sphere", sph, "diffuse", 1, kappa=1.0),
        case("sphere, weno5", sph, "advect", 3, scheme=WENO(5)),
        case("sphere", sph, "advect_momentum", 2, 2),
        Case(
            "masked_laplacian",
            lambda h: masked_laplacian(h, f2, 1.0, 1.0),
            1,
            s2,
        ),
    ]
    return C


# -----------------------------------------------------------------------------


def _timeit(f, args, min_time):
    out = f(*args)
    jax.block_until_ready(out)
    jax.block_until_ready(f(*args))
    ts, t0 = [], time.perf_counter()
    while len(ts) < 5 or (time.perf_counter() - t0 < min_time and len(ts) < 500):
        t = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t)
    return statistics.median(ts)


def _hlo(f, args):
    c = f.lower(*args).compile()
    txt = c.as_text()
    try:
        temp = int(c.memory_analysis().temp_size_in_bytes)
    except Exception:  # pragma: no cover
        temp = -1
    return len(re.findall(r"=\s*\S+\s+fusion\(", txt)), temp


def _n_slices(fn, args):
    jaxpr = jax.make_jaxpr(fn)(*args)

    def walk(j):
        n = 0
        for e in j.eqns:
            if e.primitive.name == "slice":
                n += 1
            for p in e.params.values():
                if hasattr(p, "jaxpr"):
                    n += walk(p.jaxpr)
        return n

    return walk(jaxpr.jaxpr)


def run(n: int, nz: int, min_time: float, only: str | None, repeat: int = 1):
    key, key2 = jax.random.key(0), jax.random.key(7)
    # roofline reference: an elementwise kernel reading 1 and writing 1 array
    ref_x = jax.random.normal(key, (n, n))
    t_ref = _timeit(jax.jit(lambda x: 2.0 * x + 1.0), (ref_x,), min_time)
    bw = 2 * ref_x.size * 4 / t_ref  # bytes / s
    rows = []
    for c in _cases(n, nz):
        if only and not re.search(only, c.name):
            continue
        ks = jax.random.split(key, c.n_in)
        args = tuple(jax.random.normal(k, c.shape) for k in ks)
        fwd = jax.jit(c.fn)
        vjp = jax.jit(lambda ct, *a, _f=c.fn: jax.vjp(_f, *a)[1](ct))
        jvp = jax.jit(lambda *at, _f=c.fn, _n=c.n_in: jax.jvp(_f, at[:_n], at[_n:])[1])
        tans = tuple(
            jax.random.normal(k, c.shape) for k in jax.random.split(key2, c.n_in)
        )
        try:
            out_shape = jax.eval_shape(c.fn, *args)
            ct = jax.tree_util.tree_map(
                lambda s: jax.random.normal(jax.random.key(99), s.shape, s.dtype),
                out_shape,
            )
            tf = min(_timeit(fwd, args, min_time) for _ in range(repeat))
            tb = min(_timeit(vjp, (ct, *args), min_time) for _ in range(repeat))
            tj = min(_timeit(jvp, (*args, *tans), min_time) for _ in range(repeat))
            ff, mf = _hlo(fwd, args)
            fb, mb = _hlo(vjp, (ct, *args))
            fj, _ = _hlo(jvp, (*args, *tans))
            ns = _n_slices(c.fn, args)
        except Exception as e:  # pragma: no cover
            print(f"!! {c.name}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        size = int(np.prod(c.shape)) * 4
        ideal_f = (c.n_in + c.n_out) * size / bw
        ideal_b = (2 * c.n_in + c.n_out) * size / bw
        row = dict(
            name=c.name,
            shape=list(c.shape),
            fwd_ms=tf * 1e3,
            vjp_ms=tb * 1e3,
            jvp_ms=tj * 1e3,
            fwd_fusions=ff,
            vjp_fusions=fb,
            jvp_fusions=fj,
            fwd_temp_mb=mf / 2**20,
            vjp_temp_mb=mb / 2**20,
            slices=ns,
            fwd_eff=ideal_f / tf,
            vjp_eff=ideal_b / tb,
        )
        rows.append(row)
        print(
            f"{c.name:44s} fwd {tf * 1e3:8.3f} ms ({row['fwd_eff']:4.0%}) {ff:2d}f {mf / 2**20:6.1f}MB | "
            f"vjp {tb * 1e3:8.3f} ms ({row['vjp_eff']:4.0%}) {fb:2d}f {mb / 2**20:6.1f}MB | "
            f"jvp {tj * 1e3:8.3f} ms {fj:2d}f | {ns:3d} slices",
            file=sys.stderr,
        )
    meta = dict(
        n=n, nz=nz, jax=jax.__version__, device=str(jax.devices()[0]), bw_gbps=bw / 1e9
    )
    return dict(meta=meta, rows=rows)


def profile(n: int, nz: int, pattern: str, reps: int = 10, top: int = 8):
    """Print the most expensive XLA kernels of the jitted VJP of each match."""
    import collections
    import glob
    import gzip
    import tempfile

    key = jax.random.key(0)
    for c in _cases(n, nz):
        if not re.search(pattern, c.name):
            continue
        args = tuple(
            jax.random.normal(k, c.shape) for k in jax.random.split(key, c.n_in)
        )
        ct = jax.tree_util.tree_map(
            lambda s: jax.random.normal(jax.random.key(99), s.shape, s.dtype),
            jax.eval_shape(c.fn, *args),
        )
        vjp = jax.jit(lambda ct, *a, _f=c.fn: jax.vjp(_f, *a)[1](ct))
        jax.block_until_ready(vjp(ct, *args))
        with tempfile.TemporaryDirectory() as tmp:
            with jax.profiler.trace(tmp):
                for _ in range(reps):
                    jax.block_until_ready(vjp(ct, *args))
            path = glob.glob(f"{tmp}/**/*.trace.json.gz", recursive=True)[0]
            with gzip.open(path) as fh:
                events = json.load(fh)["traceEvents"]
        kernels = collections.Counter()
        for e in events:
            name = e.get("name", "")
            is_kernel = "fusion" in name or name.split(".")[0] in (
                "concatenate",
                "pad",
                "slice",
                "copy",
                "transpose",
            )
            if e.get("ph") == "X" and is_kernel and not name.startswith("end:"):
                kernels[name] += e.get("dur", 0)
        total = sum(kernels.values()) / reps / 1e3
        print(f"{c.name}: VJP kernels {total:.3f} ms")
        for name, dur in kernels.most_common(top):
            print(f"  {dur / reps / 1e3:8.3f} ms  {name}")


def compare(a_path: str, b_path: str):
    A = {r["name"]: r for r in json.load(open(a_path))["rows"]}
    B = {r["name"]: r for r in json.load(open(b_path))["rows"]}
    print(
        "| operator | fwd before → after (ms) | speedup "
        "| vjp before → after (ms) | speedup "
        "| jvp before → after (ms) | speedup | vjp fusions | slices |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for k in A:
        if k not in B:
            continue
        a, b = A[k], B[k]
        print(
            f"| `{k}` | {a['fwd_ms']:.3f} → {b['fwd_ms']:.3f} | {a['fwd_ms'] / b['fwd_ms']:.2f} "
            f"| {a['vjp_ms']:.3f} → {b['vjp_ms']:.3f} | {a['vjp_ms'] / b['vjp_ms']:.2f} "
            f"| {a['jvp_ms']:.3f} → {b['jvp_ms']:.3f} | {a['jvp_ms'] / b['jvp_ms']:.2f} "
            f"| {a['vjp_fusions']} → {b['vjp_fusions']} | {a['slices']} → {b['slices']} |"
        )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--n", type=int, default=512, help="2-D grid size incl. ghost ring")
    p.add_argument("--nz", type=int, default=10, help="3-D levels incl. ghost slabs")
    p.add_argument("--min-time", type=float, default=0.3)
    p.add_argument("--only", default=None, help="regex filter on operator names")
    p.add_argument(
        "--repeat", type=int, default=1, help="time each case k times, keep the min"
    )
    p.add_argument("--out", default=None)
    p.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    p.add_argument(
        "--profile", metavar="REGEX", help="print the top XLA kernels of the VJP"
    )
    a = p.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    if a.profile:
        profile(a.n, a.nz, a.profile)
        return
    res = run(a.n, a.nz, a.min_time, a.only, a.repeat)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
