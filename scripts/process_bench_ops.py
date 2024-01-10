import sys


def get_options(line: str, mode: str):
    entries = line.split("-")
    if mode == "random":
        # 1000000-1500-128-pyg-float32-backward
        k = entries[2].strip()
        dataset = entries[1].strip()
        engine = entries[3].strip()
        phase = entries[5].strip()
    else:
        # 32-AIFB-slices0-fasten-float32-forward
        k = entries[-1].strip()
        dataset = entries[1].strip()
        engine = entries[3].strip()
        phase = entries[5].strip()
    return k, dataset, engine, phase


def get_minimum_time(line: str):
    entries = line.split(",")
    return entries[5].strip()


def get_maximum_time(line: str):
    entries = line.split(",")
    return entries[6].strip()


file = sys.argv[1]
if len(sys.argv) >= 3:
    mode = sys.argv[2]
perf = dict()

# read it line by line
with open(file) as f:
    lines = f.readlines()
    line_idx = 0
    while line_idx < len(lines):
        if "-" in lines[line_idx]:
            k, dataset, engine, phase = get_options(lines[line_idx], mode)
            key = f"{k}-{dataset}-{engine}-{phase}"
            line_idx += 1
            if engine == "fasten":
                if phase == "forward":
                    # single kernel, get the minimum time
                    time = get_minimum_time(lines[line_idx])
                    line_idx += 1
                    perf[key] = time
                elif phase == "backward":
                    time_dx = get_minimum_time(lines[line_idx])
                    line_idx += 2  # skip forward warmup
                    time_dw = get_minimum_time(lines[line_idx])
                    line_idx += 1
                    perf[key] = int(time_dx) + int(time_dw)
                else:
                    raise Exception("Invalid phase: " + phase)
            elif engine == "pyg":
                if phase == "forward":
                    # single kernel, get the minimum time
                    time = get_minimum_time(lines[line_idx])
                    line_idx += 1
                    perf[key] = time
                elif phase == "backward":
                    time_dx = get_minimum_time(lines[line_idx])
                    time_dw = get_maximum_time(lines[line_idx])
                    line_idx += 1
                    perf[key] = int(time_dx) + int(time_dw)
                else:
                    raise Exception("Invalid phase: " + phase)
            else:
                raise Exception("Invalid engine: " + engine)
        else:
            raise Exception("Invalid line: " + lines[line_idx])

print("perf = ", perf)
