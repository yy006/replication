import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class DataAnalyzer:
    def __init__(self, data_source, environment='normal'):
        # data_sourceが文字列の場合、ファイルパスとして扱う
        if isinstance(data_source, str):
            if environment == 'colab':
                # Google Colab環境の場合のファイルパス処理
                from google.colab import drive
                drive.mount('/content/drive')
                file_path = os.path.join('/content/drive/My Drive/', data_source)
            else:
                # 通常のPython環境の場合のファイルパス処理
                file_path = data_source
            
            self.data = pd.read_csv(file_path)
        elif isinstance(data_source, pd.DataFrame):
            # data_sourceがDataFrameの場合、そのまま使用
            self.data = data_source
        else:
            raise ValueError("data_source must be a filepath string or a DataFrame")
        
        # 初期分析を実行
        self.initial_analysis()

    def initial_analysis(self):
        # データの最初の5行を表示
        print(self.data.head())
        # データの情報を表示（データ型、欠損値など）
        print(self.data.info())
        # 数値データの統計的要約を表示
        print(self.data.describe())
        # カテゴリデータのユニークな値とその出現回数を表示
        for col in self.data.select_dtypes(include=['object']).columns:
            print(f'Unique values for {col}:')
            print(self.data[col].value_counts())

    def show_missing_values(self):
        # 各列の欠損値の数を表示
        missing_values = self.data.isnull().sum()
        print(missing_values[missing_values > 0])

    def show_feature_info(self, feature_name):
        # 特定の特徴量の情報を表示
        if feature_name in self.data.columns:
            print(f'Data type of {feature_name}: {self.data[feature_name].dtype}')
            print(f'Statistics of {feature_name}:')
            print(self.data[feature_name].describe())
            if self.data[feature_name].dtype == 'object':
                print(f'Unique values of {feature_name}:')
                print(self.data[feature_name].value_counts())
        else:
            print(f'{feature_name} is not a valid column name.')

    def analyze_datetime(self, datetime_column):
        '''
        各日時のデータ数を返すメソッド
        :datetime_column: 日時データの列名
        '''
        # datetime_columnを日時型に変換し、インデックスに設定
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        self.data.set_index(datetime_column, inplace=True)

        # 日ごとのデータポイント数を計算
        daily_counts = self.data.resample('D').size()

        # 時間ごとのデータポイント数を計算
        hourly_counts = self.data.resample('H').size()

        # 日ごとのデータポイント数をテキストで出力
        daily_summary = '\n'.join([f'{date}: {count}' for date, count in daily_counts.items()])
        print("日ごとのデータポイント数:\n", daily_summary)

        # すべての日における時間ごとのデータポイント数をテキストで出力
        hourly_summary = '\n'.join([f'{time}: {count} ' for time, count in hourly_counts.items()])
        print("\n時間ごとのデータポイント数（すべての日）:\n", hourly_summary)


    def analyze_datetime_graph(self, datetime_column):
        '''
        時間ごとのデータを折れ線グラフでプロット
        :datetime_column: 日時データの列名
        '''
        # datetime_columnを日時型に変換し、インデックスに設定
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        self.data.set_index(datetime_column, inplace=True)

        # 日ごとのデータポイント数を計算
        daily_counts = self.data.resample('D').size()

        # 時間ごとのデータポイント数を計算
        hourly_counts = self.data.resample('H').size()

        # 時間ごとのデータポイント数を折れ線グラフでプロット
        plt.figure(figsize=(15, 5))
        plt.plot(hourly_counts.index, hourly_counts.values, marker='o', linestyle='-')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
        plt.xticks(rotation=90)
        plt.title('時間ごとのデータポイント数')
        plt.xlabel('時間')
        plt.ylabel('データポイント数')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def analyze_and_sort_by_label(self, column_names=['class', 'attackType', 'attackID'], top_n_ips=20, select_IP='Src IP Addr'):
        '''
        指定した列について、ラベルごとの出現回数の多いSrc IP Addrを見る
        top_n_ips: 各ラベルごとの上位IPアドレス数
        '''
        # ローカルアドレス（192.168で始まるアドレス）を除外
        # self.data = self.data[~self.data[select_IP].str.startswith('192.168')]

        # 各列について処理
        for col in column_names:
            if col in self.data.columns:
                # ラベルのカウント
                label_counts = self.data[col].value_counts()

                # ラベルごとのSrc IP Addrのリストを作成
                sorted_ips_by_label = {}
                for label in label_counts.index:
                    sorted_ips_counts = self.data[self.data[col] == label][select_IP].value_counts().head(top_n_ips)
                    sorted_ips_by_label[label] = sorted_ips_counts.to_dict()

                # 結果を表示
                print(f"Column '{col}' label counts:\n{label_counts}\n")
                print(f"Top {top_n_ips} sorted Src IP Addr for each label in '{col}':\n")
                for label, ips in sorted_ips_by_label.items():
                    print(f"Label: {label}")
                    for ip, count in ips.items():
                        print(f"  {ip}: {count}")
                    print()

            else:
                print(f"Column '{col}' does not exist in the DataFrame.")

    def analyze_and_sort_ips_by_frequency_ratio(self, column_names=['class', 'attackType', 'attackID'], top_n_ips=20):
        # 各列について処理
        for col in column_names:
            if col in self.data.columns:
                # 全体の登場回数
                total_counts = self.data[col].value_counts().sum()

                # ラベルごとのSrc IP Addrの割合を計算
                ratio_ips_by_label = {}
                for label in self.data[col].unique():
                    label_counts = self.data[self.data[col] == label]['Src IP Addr'].value_counts()
                    ratio = label_counts / total_counts
                    sorted_ratio = ratio.sort_values(ascending=False).head(top_n_ips)
                    ratio_ips_by_label[label] = sorted_ratio

                # 結果を表示
                print(f"Column '{col}':")
                for label, ips in ratio_ips_by_label.items():
                    print(f"Label: {label}")
                    for ip, ratio in ips.items():
                        print(f"  {ip}: {ratio:.2f}")
                    print()

            else:
                print(f"Column '{col}' does not exist in the DataFrame.")

# 使用例
# analyzerは、データフレームが格納されているクラスのインスタンスです。
# analyzer.analyze_and_sort_by_label()


# 使用例
# file_path = 'CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week4.csv'  # CSVファイルのパスを指定
# 以下のファイルをcsvに変換する
file_path = 'CTU-13_Scenario-50/capture20110817.pcap.netflow.labeled.csv'  # CSVファイルのパスを指定
analyzer = DataAnalyzer(file_path)
# analyzer.show_missing_values()  # 欠損値の確認
# analyzer.show_feature_info('feature_name')  # 特定の特徴量の情報を表示

# analyzer.analyze_datetime_graph('Date first seen')  # 日時データの分析とプロット
analyzer.analyze_and_sort_by_label(select_IP='Src IP Addr') # 指定した列について、ラベルごとのデータ数と出現回数の多いデータを返す
#analyzer.analyze_and_sort_ips_by_frequency_ratio()

'''
# 使用例
# 通常のPython環境での使用
file_path = 'path_to_your_csv_file.csv'
analyzer = DataAnalyzer(file_path)

# Google Colabでの使用
# file_pathはGoogle Drive内のファイルパスにする
file_path = 'path_to_your_csv_file.csv'
analyzer = DataAnalyzer(file_path, environment='colab')

# DataFrameを直接渡す場合
df = pd.read_csv('path_to_your_csv_file.csv')
analyzer = DataAnalyzer(df)
'''
