"""Step 2a: build the landmark dataset with pandas and print the cohort size per landmark."""
import argparse
import config
from features import load_visits, build_landmark_pandas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-cdr", type=float, default=0.5)
    a = ap.parse_args()
    visits = load_visits(config.VISITS_CSV)
    print(f"{'landmark':>8} {'subjects':>9} {'progressed':>11} {'stable':>7}")
    for L in a.landmarks:
        d = build_landmark_pandas(visits, L, a.max_cdr)
        d.to_csv(config.OUT_DIR / f"landmark{L}.csv", index=False)
        pos = int(d["progressed"].sum()) if len(d) else 0
        print(f"{L:>8} {len(d):>9} {pos:>11} {len(d) - pos:>7}")
    print(f"\nSaved landmark*.csv in {config.OUT_DIR}")


if __name__ == "__main__":
    main()
