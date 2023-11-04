import pandas as pd

class DataAnalyzer:
    def __init__(self, file_path):
        # CSVファイルを読み込む
        self.data = pd.read_csv(file_path)
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

# 使用例
file_path = 'CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv'  # CSVファイルのパスを指定
analyzer = DataAnalyzer(file_path)
analyzer.show_missing_values()  # 欠損値の確認
analyzer.show_feature_info('feature_name')  # 特定の特徴量の情報を表示
