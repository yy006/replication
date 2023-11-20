import pandas as pd
import os

class DataPreprocessor:
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
        
    def preprocess(self, num_rows=None, features=None, start_date=None):
        """
        データの前処理を行うメソッド
        :param num_rows: 取得する行数
        :param features: 特徴量のリスト（列名）
        :param start_date: 取得開始日（YYYY-MM-DD形式の文字列）
        :return: 前処理されたDataFrame
        """
        df = self.data

        # 日付に基づいてフィルタリング
        if start_date is not None:
            df['Date first seen'] = pd.to_datetime(df['Date first seen'])
            start_date = pd.to_datetime(start_date)
            df = df[df['Date first seen'] >= start_date]
            print(df)

        # 行数の制限
        if num_rows is not None:
            df = df.head(num_rows)

        # 特徴量に基づいて列を並べ替え
        if features is not None:
            existing_features = [feature for feature in features if feature in df.columns]
            df = df[existing_features]

        return df

'''
# 使用例
# ファイルパスまたはDataFrameを渡してインスタンスを作成
preprocessor = DataPreprocessor('dataset/CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv')

# 特定の特徴量を指定して前処理を実行（例：先頭10行）
features_to_include = ['Src IP Addr', 'Dst IP Addr', 'Proto', 'Src Pt', 'Dst Pt', 'class', 'attackType', 'attackID', 'attackDescription']
processed_df = preprocessor.preprocess(num_rows=10, features=features_to_include, start_date="2017-03-16 00:00:00")

print(processed_df)
'''