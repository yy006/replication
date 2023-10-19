from scapy.all import rdpcap

# pcapファイルを読み込む
packets = rdpcap('botnet-capture-20110817-bot.pcap')

# 先頭の10行を表示
for i, packet in enumerate(packets[:10]):
    print(f"Packet #{i+1}:")
    packet.show()

# 列の名前を表示
# 実際には、pcapファイルには「列」という概念はありませんが、
# 各パケットのレイヤーとフィールドを表示することができます。
# ここでは、最初のパケットのレイヤーとフィールドを示しています。
if packets:
    print("\nFields (Columns):")
    for field in packets[0].fields:
        print(field)
