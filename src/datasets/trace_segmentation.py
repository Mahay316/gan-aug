from scapy.all import *
from scapy.layers.inet import TCP, UDP


def extract_packet_info(pkt):
    if TCP in pkt:
        transport_proto = 'TCP'
    elif UDP in pkt:
        transport_proto = 'UDP'
    else:
        return ()

    proto = pkt['IP'].proto
    src = pkt['IP'].src
    dst = pkt['IP'].dst
    sport = pkt[transport_proto].sport
    dport = pkt[transport_proto].dport

    if src > dst:
        direction = 1
        conn = (proto, src, dst, sport, dport)
    else:
        direction = -1
        conn = (proto, dst, src, dport, sport)

    data = [float(pkt.time), int(direction * len(pkt))]
    return conn, data


def trim_and_padding(segment, upperbound, padding=None):
    if padding is None:
        padding = [0, 0, 0]

    target_len = 0
    for i in range(len(segment)):
        if len(segment[i]) > upperbound:
            segment[i] = segment[i][:upperbound]

        target_len = max(target_len, len(segment[i]))

    for i in range(len(segment)):
        if len(segment[i]) < target_len:
            padding_len = target_len - len(segment[i])
            segment[i] += [padding] * padding_len


def write_sample(segment, label, output):
    trim_and_padding(segment, 1000)

    json.dump([segment, label], output)
    output.write('\n')


def long_enough(segment, threshold) -> bool:
    if len(segment) == 0:
        return False

    avg_flow_len = sum([len(flow) for flow in segment]) // len(segment)
    print(f'{avg_flow_len}, {len(segment)} {"<<drop>>" if avg_flow_len < threshold else ""}')
    return avg_flow_len >= threshold


def packet2vector(filename, output, label, time_thresh, len_thresh, mode: str):
    first_start_flag = True

    # variables prefixed with stat_ are for statistics purposes
    stat_segment_cnt = 0

    segment = []
    flows = {}

    start_time = 0
    end_time = 0

    with PcapReader(filename) as reader:
        for pkt in reader:
            pkt_info = extract_packet_info(pkt)
            if len(pkt_info) < 2:
                continue

            conn = pkt_info[0]
            data = pkt_info[1]
            if first_start_flag:
                first_start_flag = False
                end_time = start_time = pkt.time

            # packet belongs to a new segment
            # hence we need to reset variables related to the previous segment
            if (mode == 'span' and pkt.time - start_time > time_thresh) \
                    or (mode == 'interval' and pkt.time - end_time > time_thresh):
                # filter out segments that are too short
                if long_enough(segment, len_thresh):
                    write_sample(segment, label, output)
                    stat_segment_cnt += 1

                start_time = pkt.time
                segment = []
                flows = {}

            # create a new flow if it doesn't exist
            if conn not in flows:
                flows[conn] = len(flows)
                segment.append([])

            # construct flows inside segment
            flow_id = flows[conn]
            prev_time = (start_time + segment[flow_id][-1][1]) if len(segment[flow_id]) > 0 else pkt.time
            pv = (
                data[1],
                float(pkt.time - start_time),
                float(pkt.time - prev_time)
            )
            # update end_time as the arrival time of the latest packet in current trace
            end_time = pkt.time
            segment[flow_id].append(pv)

    # save the last segment after the loop ends
    if long_enough(segment, len_thresh):
        write_sample(segment, label, output)


if __name__ == '__main__':
    type2label = {
        'nontor': 0,
        'tor': 1,
        'obfs4': 2,
        'webtunnel': 3,
        'snowflake': 4,
        'dnstt': 5,
        'shadowsocks': 6
        # 'conjure': 7,
        # 'cloak': 8,
    }

    # entry point
    if len(sys.argv) < 4:
        print(f'Usage: {sys.argv[0]} mode time_thresh label')
        print(f'Possible values for mode: [span, interval]')
        print(f'Possible values for label:', list(type2label.keys()))
        exit(-1)

    mode = sys.argv[1]
    time_thresh = float(sys.argv[2])  # in seconds
    label = int(type2label[sys.argv[3]])

    output_file = f'pt_{sys.argv[3]}_{mode}.txt'
    record_file = f'./record_{mode}.tmp'

    with open(record_file, 'a+') as rec:
        rec.seek(0)
        processed_file = {line.strip() for line in rec}

        for file in os.listdir('.'):
            if file.endswith('.pcap') and file not in processed_file:
                # packet vectorization
                print('Processing pcap file:', file)
                with open(output_file, 'a') as output:
                    packet2vector(file, output, label, time_thresh, 32, mode)
                    output.flush()

                # processed file bookkeeping
                processed_file.add(file)
                rec.write(file + '\n')
                rec.flush()
