import csv
'''
with open('CTU-13_Scenario-50/capture20110817.pcap.netflow - コピー.txt', 'r') as file, open('output.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    for line in file:
        parts = line.split()
        writer.writerow(parts)  # 分割された部分をCSVファイルに書き込みます
'''
'''
import pandas as pd
df = pd.read_csv('output.csv')

df = df.rename(columns={
    'flow': 'flow_start',
    'start': 'Durat',
    'Durat': 'Prot',
    'Prot': 'SrcIP_Addr_Port',
    'IP': 'DstIP_Addr_Port',
    'Addr:Port': 'Flags',
    'Dst': 'Tos',
    'IP.1': 'Packets',
    'Addr:Port.1': 'Bytes',
    'Flags': 'Flows',
    'Tos': 'Label'
})

df = df.drop(columns=['Src'])  # IP列を削除

# 'Label'列以降の5列を削除
df = df.iloc[:, :-5]

# 新しいCSVファイルに保存
df.to_csv('new_output.csv', index=False)
print(df)
'''

import pandas as pd
df = pd.read_csv('capture20110817.pcap.netflow.labeled.csv')

# 'SrcIP_Addr_Port'列を分割
df[['Src IP Addr', 'Src Pt']] = df['SrcIP_Addr_Port'].str.split(':', n=1, expand=True)

# 'DstIP_Addr_Port'列を分割
df[['Dst IP Addr', 'Dst Pt']] = df['DstIP_Addr_Port'].str.split(':', n=1, expand=True)

df = df.drop(columns=['SrcIP_Addr_Port', 'DstIP_Addr_Port']) # 列を削除

df = df.rename(columns={
    'Prot': 'Proto',
    'Label': 'class',
})

# 新しいCSVファイルに保存
df.to_csv('new_output.csv', index=False)
print(df)
'''
df = df.rename(columns={
    'flow': 'flow_start',
    'start': 'Durat',
    'Durat': 'Prot',
    'Prot': 'SrcIP_Addr_Port',
    'IP': 'DstIP_Addr_Port',
    'Addr:Port': 'Flags',
    'Dst': 'Tos',
    'IP.1': 'Packets',
    'Addr:Port.1': 'Bytes',
    'Flags': 'Flows',
    'Tos': 'Label'
})

'Src IP Addr', 'Dst IP Addr', 'Proto', 'Src Pt', 'Dst Pt',
'''