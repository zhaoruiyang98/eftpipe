ref = {
    "omch2": {"mean": 0.1369, "upper": 0.011, "lower": 0.016,},
    "H0": {"mean": 68.88, "upper": 1.4, "lower": 1.8,},
    "logA": {"mean": 2.834, "upper": 0.2, "lower": 0.21},
    "ns": {"mean": 0.9134, "upper": 0.077, "lower": 0.074,},
    "mnu": {"mean": 0.3901, "upper": 0.11, "lower": 0.33,},
    "sigma8": {"mean": 0.7217, "upper": 0.045, "lower": 0.05,},
}


def main():
    dct = {}
    for k, _ in ref.items():
        str = input(f"Please enter the mean, lower 68%, upper 68% of <{k}>:\n")
        received = [float(x) for x in str.split()]
        print(
            f"received: mean = {received[0]}, "
            f"lower 68% = {received[1]}, "
            f"upper 68% = {received[2]}"
        )
        dct[k] = {
            "mean": received[0],
            "upper": received[2] - received[0],
            "lower": received[0] - received[1],
        }
    header = f"{'name'.rjust(8)}{'mean'.rjust(8)}{'+'.rjust(8)}{'-'.rjust(8)}{'diff'.rjust(8)}"
    template = "{name:>8}{mean:>8.4f}{up:>8.3f}{low:>8.3f}{diff:>8.2f}"
    print("=" * len(header))
    print(header)
    for k, v in ref.items():
        sigma = 0.5 * (v["upper"] + v["lower"])
        diff = (dct[k]["mean"] - ref[k]["mean"]) / sigma
        print(
            template.format(
                name=k,
                mean=dct[k]["mean"],
                up=dct[k]["upper"],
                low=dct[k]["lower"],
                diff=diff,
            )
        )


if __name__ == "__main__":
    main()
