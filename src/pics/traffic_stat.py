import json


def get_packet_count(traffic_filename: str) -> list[list[int]]:
    # each element represents a trace
    # consisting of packet count for each flow in this specific trace
    # e.g. [[3, 5, 3], [3, 2], [22, 33, 1, 5], ...]
    stat_packet_cnt = []

    with open(traffic_filename, 'r') as f:
        for line in f:
            trace = json.loads(line)[0]

            packet_cnt_per_flow = [len(i) for i in trace]
            stat_packet_cnt.append(packet_cnt_per_flow)

    return stat_packet_cnt


def get_byte_count(traffic_filename: str) -> list[list[int]]:
    stat_byte_cnt = []

    with open(traffic_filename, 'r') as f:
        for line in f:
            trace = json.loads(line)[0]

            byte_cnt_per_flow = [sum(abs(p[0]) for p in flow) for flow in trace]
            stat_byte_cnt.append(byte_cnt_per_flow)

    return stat_byte_cnt
