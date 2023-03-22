import json
import os
import pickle
import sys

from datetime import datetime

def get_input(local=False):
    if local:
        print("Reading local file.")

        return "file0"

    dids = os.getenv("DIDS", None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    for did in dids:
        filename = f"/data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")

        return filename


def run_algorithm(local=False):
    hsma = 40
    lsma = 12

    filename = get_input(local)
    if not filename:
        print("Could not retrieve filename.")
        return

    with open(filename) as datafile:
        data = datafile.readlines(0)[0]
        data = json.loads(data)

    uts = [int(xi[0]) / 1000 for xi in data]

    # Data is per hour so reformat to per day
    ts_obj = [datetime.utcfromtimestamp(s) for s in uts]
    eth = [float(xi[4]) for xi in data]
    dataset = pd.DataFrame({"y": eth}, index=ts_obj)
    resampled = dataset.resample("1D").last()

    assert len(resampled) >= hsma, f"Not enough data for windows {len(resampled)} vs {hsma}"

    eth = resampled["y"]

    # Compute SMAs from the past sample
    hsma_day_before = sum(eth[-hsma - 1:-1]) / len(eth[-hsma - 1:-1])
    lsma_day_before = sum(eth[-lsma - 1:-1]) / len(eth[-lsma - 1:-1])

    # Compute SMAs from the current sample
    hsma_current_day = sum(eth[-hsma:]) / len(eth[-hsma:])
    lsma_current_day = sum(eth[-lsma:]) / len(eth[-lsma:])

    # Detect Cross
    day_before_diff = lsma_day_before - hsma_day_before
    current_day_diff = lsma_current_day - hsma_current_day

    # Golden Cross
    if day_before_diff < 0 and current_day_diff > 0:
        advice = "buy"
        # Death Cross
    elif day_before_diff > 0 and current_day_diff < 0:
        advice = "sell"
        # No Cross
    else:
        advice = "do nothing"


    filename = "advice.pkl" if local else "/data/outputs/result"
    with open(filename, "+wb") as pickle_file:
        print(f"Pickling results in {filename}")
        pickle.dump(advice, pickle_file)


if __name__ == "__main__":
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_algorithm(local)
