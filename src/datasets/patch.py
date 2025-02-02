import json

filename = 'D:/BaiduNetdiskDownload/Tor Traffic/pt_tor.txt'
filename_out = 'D:/BaiduNetdiskDownload/Tor Traffic/pt_tor_bk.txt'
with open(filename, 'r') as f, open(filename_out, 'w', encoding='utf-8', newline='\n') as fo:
    for line in f:
        sample = json.loads(line)
        sample[1] = 5
        fo.write(json.dumps(sample) + '\n')
