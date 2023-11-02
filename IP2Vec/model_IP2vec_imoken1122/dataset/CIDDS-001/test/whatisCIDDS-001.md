1 Src IP Source : IP Address
2 Src Port : Source Port
3 Dest IP Destination IP Address
4 Dest Port Destination Port
5 Proto Transport Protocol (e.g. ICMP, TCP, or UDP)
6 Date first seen Start time flow first seen
7 Duration Duration of the flow
8 Bytes Number of transmitted bytes
9 Packets Number of transmitted packets
10 Flags OR concatenation of all TCP Flags
11 Class Class label (normal, attacker, victim, suspicious or
unknown)
12 AttackType Type of Attack (portScan, dos, bruteForce, —)
13 AttackID Unique attack id. All flows which belong to the
same attack carry the same attack id.
14 AttackDescription Provides additional information about the set attack parameters (e.g. the number of attempted
password guesses for SSH-Brute-Force attacks