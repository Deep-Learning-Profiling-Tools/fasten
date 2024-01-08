import sys


def get_options(line: str):
    # 32-AIFB-slices0-fasten-float32-forward
    entries = line.split("-")
    k = entries[0]
    dataset = entries[1]
    engine = entries[3]
    phase = entries[5]
    return k, dataset, engine, phase


def get_minimum_time(line: str):
    entries = line.split(",")
    return entries[5]


def get_maximum_time(line: str):
    entries = line.split(",")
    return entries[6]


file = sys.argv[1]
perf = dict()

# read it line by line
with open(file) as f:
    lines = f.readlines()
    line_idx = 0
    while line_idx < len(lines):
        if "-" in lines[line_idx]:
            k, dataset, engine, phase = get_options(lines[line_idx])
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
                    perf[key] = time_dx + time_dw
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
                    perf[key] = time_dx + time_dw
                else:
                    raise Exception("Invalid phase: " + phase)
            else:
                raise Exception("Invalid engine: " + engine)
        else:
            raise Exception("Invalid line: " + lines[line_idx])

print("perf = ", perf)
