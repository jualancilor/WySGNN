"""Fetch the JARVIS-DFT piezoelectric tensor dataset (optional).

Requires ``jarvis-tools``.  Downloads the ``dft_3d`` database, keeps records with
a valid ``piezoelectric_tensor`` (Voigt 3x6, units C/m^2), and caches them to
``results/jarvis_piezo.json`` for use with ``source="jarvis"``.

If this fails (no network / package), the pipeline falls back to the synthetic
teacher dataset, which requires no download.
"""

from __future__ import annotations

import json
import os
import sys


def main():
    try:
        from jarvis.db.figshare import data as jdata
    except Exception:
        print("jarvis-tools not installed; run `pip install jarvis-tools`. "
              "The pipeline works without it via source='synthetic'.")
        sys.exit(1)

    print("[fetch] downloading JARVIS dft_3d (this is large)...")
    records = jdata("dft_3d")
    kept = []
    for r in records:
        pt = r.get("piezoelectric_tensor")
        if isinstance(pt, (list, tuple)) and len(pt) == 3 and len(pt[0]) == 6:
            kept.append({"jid": r.get("jid"), "spg_number": r.get("spg_number"),
                         "piezoelectric_tensor": pt, "atoms": r.get("atoms")})
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "jarvis_piezo.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(kept, open(out, "w"))
    print(f"[fetch] kept {len(kept)} records with valid piezoelectric tensors -> {out}")


if __name__ == "__main__":
    main()
