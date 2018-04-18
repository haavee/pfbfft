import argparse

# take in some command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("--debug", default=lambda *args: None, action='store_const', const=print)
ap.add_argument("--lo", type=float, default=0)
ap.add_argument("--bw", type=float, default=8e6)
ap.add_argument("--ftsize", type=int, default=1024)
ap.add_argument("--nspec", type=int, default=10)
ap.add_argument("--window", type=str, default='hamming')
ap.add_argument("--peak-threshold", type=float, default=0.9, help="Peak detection threshold (fraction of max)")

globals().update(ap.parse_args().__dict__)
