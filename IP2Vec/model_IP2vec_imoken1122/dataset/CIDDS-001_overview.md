## CIDDS-001 とは
[CIDDS-001.pdf](https://www.hs-coburg.de/fileadmin/hscoburg/Forschung/Flow-based_benchmark_data_sets_for_intrusion_detection.pdf)
- [OpenStack](https://www.openstack.org/)を使用して小規模ビジネス環境をエミュレートしたもの。
- この環境には、複数のクライアントと、Eメール,Webサーバーのような典型的なサーバーが含まれる。
- CIDDS-001には、一方向のNetFlowデータが含まれている

## エミュレートしたネットワーク環境
- 図は、エミュレートされた小規模ビジネス環境の概要を示している
- 星印はトラフィックがキャプチャされた場所
- エミュレートされた中小企業環境は 4 つのサブネットで構成されている。 1つのサブネットにはすべての内部サーバー（ウェブ、ファイル、バックアップ、メール）が含まれる。
- 各サブネットの概要

## CIDDS-001データセットの公開データ
- 攻撃ログ
    - OpenStack環境内で実行された攻撃、外部サーバーに対する攻撃の二種類が含まれている
    - 攻撃元IPアドレス、攻撃開始時刻、攻撃終了時刻、攻撃の簡単な説明 の情報が記述されている (BruteForce, PortScan)

## ラベリングと匿名化
### ラベリング
各フローに4つのラベル属性（class、attackID、attackType、attackDescription）を付加する。  
class：normal, attacker(用意した攻撃者による攻撃の通信), victim, suspicious(それ以外の通信), unknown(80,443番の通信)

attackType：  
dos  
portScan  
pingScan      
bruteForce   

attackID：各攻撃に番号を振っている

attackDescription：

- OpenStack環境内のトラフィック
- 外部サーバーのトラフィック
    - 三台の攻撃者サーバ

### 匿名化

## トラフィックの特徴
- CIDDS-001のデータセットは4週間にわたってキャプチャされ、約3200万フローが含まれている
- このうち、約 3,100 万フローが OpenStack 環境内でキャプチャされ、約 0.7 百万フローが外部サーバーでキャプチャされた。
- 70件の攻撃はOpenStack環境内で実行され、22件の攻撃は外部サーバーを標的としていた
