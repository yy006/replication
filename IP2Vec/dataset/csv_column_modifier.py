import pandas as pd

def modify_csv(input_filename, output_filename, drop_columns, convert_to_int_columns):
    # CSVファイルを読み込む
    df = pd.read_csv(input_filename)

    # 指定された列を削除
    df.drop(columns=drop_columns, inplace=True)

    # 指定された列の値を整数に変換
    for column in convert_to_int_columns:
        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')

    # 結果を新しいCSVファイルに保存
    df.to_csv(output_filename, index=False)

    print(f"CSV file has been modified and saved as {output_filename}")

# 実行
input_file = 'botnet-capture-20110817-bot.csv'
output_file = 'modified_botnet.csv'
columns_to_drop = ['No.', 'Time', 'Length', 'Info']
columns_to_convert = ['Source Port', 'Destination Port']

modify_csv(input_file, output_file, columns_to_drop, columns_to_convert)
