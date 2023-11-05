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
        # self.initial_analysis()

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
        # datetime_columnを日時型に変換し、インデックスに設定
        self.data[datetime_column] = pd.to_datetime(self.data[datetime_column])
        self.data.set_index(datetime_column, inplace=True)

        # 日ごとのデータポイント数を計算
        daily_counts = self.data.resample('D').size()

        # 時間ごとのデータポイント数を計算
        hourly_counts = self.data.resample('H').size()

        # 日ごとのデータポイント数をプロット
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(daily_counts.index, daily_counts.values)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.ticklabel_format(style='plain', axis='y')  # 科学的表記法を無効にする
        plt.title('日ごとのデータポイント数')
        plt.xlabel('日付')
        plt.ylabel('データポイント数')
        plt.xticks(rotation=90)  # 日付ラベルを回転して表示
        plt.show()

        # 特定の日の時間ごとのデータポイント数をプロット（例：最初の日）
        specific_day = self.data.index.date[0]
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.bar(hourly_counts[str(specific_day)].index, hourly_counts[str(specific_day)].values)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.ticklabel_format(style='plain', axis='y')
        plt.title(f'{specific_day}の時間ごとのデータポイント数')
        plt.xlabel('時間')
        plt.ylabel('データポイント数')
        plt.xticks(rotation=90)
        plt.show()



# 使用例
file_path = 'CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv'  # CSVファイルのパスを指定
analyzer = DataAnalyzer(file_path)
analyzer.show_missing_values()  # 欠損値の確認
# analyzer.show_feature_info('feature_name')  # 特定の特徴量の情報を表示

# analyzer.analyze_datetime('Date first seen')  # 日時データの分析とプロット

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