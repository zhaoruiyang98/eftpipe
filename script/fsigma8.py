from __future__ import annotations
import argparse


def main():
    parser = argparse.ArgumentParser(
        "fsigma8", description="transform cosmo params to fsigma8"
    )
    parser.add_argument("input_yaml", type=str)
    parser.add_argument("z", type=float, nargs="+")
    parser.add_argument("-o", "--output", required=True, help="output file name")
    args = parser.parse_args()

    from eftpipe.analysis import EFTLikeProducts, cosmo_to_fsigma8

    p = EFTLikeProducts.from_yaml_file(args.input_yaml)
    cosmo = p.cosmo_model()
    cosmo_to_fsigma8(args.z, cosmo, p.chains, args.output)


if __name__ == "__main__":
    main()
