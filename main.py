import cv2
import numpy as np
import math
import os
import pandas as pd
from openpyxl.styles import PatternFill

def calculate_t_star(time, rmax):
    """
    t*を計算する関数
    time: 経過時間（s）
    rmax: 最大半径（mm）
    """
    pressure_term = (10**5/1000)**(-0.5)  # 圧力項の計算
    denominator = 0.91468 * rmax * pressure_term * 1000
    return time / denominator

def find_bubble_points(radius_data):
    """
    気泡の最大半径点と最初の極小点（崩壊点）を見つける
    """
    max_radius = 0
    max_index = -1
    collapse_index = -1

    # まず気泡の発生（0から非0への変化）を見つける
    started = False
    start_bubble_index = -1
    for i, radius in enumerate(radius_data):
        if not started and radius > 0:
            started = True
            start_bubble_index = i
        if started:
            # 最大半径を更新
            if radius > max_radius:
                max_radius = radius
                max_index = i

    # 最大半径点以降の最初の極小値を見つける
    # ただし、極小値を探す範囲を最大半径点以降に限定
    radius_threshold = 1
    if max_index != -1:
        # 最大半径点からデータ終了までを探索
        for i in range(max_index + 1, len(radius_data)):
            # 谷底を探す条件: 現在の値が両隣よりも小さい
            # ただし、配列の境界チェックも必要
            if i > 0 and i < len(radius_data) - 1:
                if radius_data[i] < radius_threshold and radius_data[i] < radius_data[i-1] and radius_data[i] <= radius_data[i+1]:
                    collapse_index = i
                    break # 最初の極小値を見つけたらループを抜ける
            elif i == len(radius_data) - 1: # 最後の要素の場合
                if radius_data[i] < radius_threshold and radius_data[i] < radius_data[i-1]:
                    collapse_index = i
                    break
    
    # もし極小値が見つからず、最後に0になる点があればそれを崩壊点とする（元のロジックの補完）
    if collapse_index == -1:
        for i in range(max_index + 1, len(radius_data)):
            if radius_data[i] == 0:
                collapse_index = i
                break
    
    return max_index, collapse_index

# wall_x_pixel を引数に追加
def calculate_bubble_properties(image_path, output_path, calibration, wall_x_pixel=None):
    # ファイルの存在確認
    if not os.path.isfile(image_path):
        print(f"画像ファイルが存在しません: {image_path}")
        return
    
    # 画像読み込み - ファイルパスを適切に処理
    try:
        # Windows環境でのファイルパス問題に対処
        # バックスラッシュを含むパスを raw 文字列としてエンコード
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            print(f"画像の読み込みに失敗しました: {image_path}")
            return
    except Exception as e:
        print(f"画像読み込み時にエラーが発生しました: {image_path} - {str(e)}")
        return

    # 画像の前処理
    blurred = cv2.GaussianBlur(img, (41, 41), 0)
    
    # 適応的閾値処理
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 29, 2)

    # モルフォロジー演算
    kernel = np.ones((9,9), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 結果の可視化用の画像を用意
    img_result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 壁面位置の描画 (wall_x_pixelが指定されている場合のみ)
    if wall_x_pixel is not None:
        # 点線を描画
        # OpenCVのlineTypeには点線タイプが直接ないので、手動で描画
        for y_coord in range(0, img.shape[0], 10): # 10ピクセル描画、10ピクセル空白の例
            cv2.line(img_result, (int(wall_x_pixel), y_coord), (int(wall_x_pixel), min(y_coord + 5, img.shape[0])), (0, 0, 255), 2) # 赤色、3ピクセル太さ

    if contours:
        # 最大の輪郭を気泡として扱う
        bubble = max(contours, key=cv2.contourArea)

        # バウンディングボックスを取得
        x, y, w, h = cv2.boundingRect(bubble)

        # 気泡のマスクを作成
        mask = np.zeros(binary.shape, dtype=np.uint8)
        cv2.drawContours(mask, [bubble], 0, 255, -1)

        volume = 0
        centers = []
        total_cx, total_cy = 0, 0
        total_area = 0

        for i in range(h):
            # スライスを取得
            slice_mask = mask[y+i, x:x+w]
            
            if np.sum(slice_mask) > 0:
                # スライスの左端と右端を見つける
                non_zero = np.nonzero(slice_mask)[0]
                left = non_zero[0]
                right = non_zero[-1]
                
                # スライスの中心を計算
                cx = x + left + (right - left) // 2
                cy = y + i

                centers.append((cx, cy))

                # スライスの面積を計算
                slice_area = np.sum(slice_mask > 0)

                # 円盤の体積を計算
                radius = (right - left) / 2
                slice_volume = math.pi * radius**2

                volume += slice_volume

                # 重心計算のための累積
                total_cx += cx * slice_area
                total_cy += cy * slice_area
                total_area += slice_area

        # 気泡全体の重心を計算
        if total_area > 0:
            center_of_mass = (int(total_cx / total_area), int(total_cy / total_area))
        else:
            center_of_mass = (x + w // 2, y + h // 2)

        # キャリブレーションを適用
        volume_mm3 = volume / (calibration**3)
        equivalent_radius_mm = (3 * volume_mm3 / (4 * math.pi))**(1/3)

        # 大きさのチェック
        min_volume = 0
        # この値は必要に応じて調整
        if volume_mm3 < min_volume:
            return {
                'volume': 0,
                'radius': 0,
                'center': (center_of_mass[0]/calibration, center_of_mass[1]/calibration)
            }

        # 図形描画 (気泡の外形と重心)
        cv2.drawContours(img_result, [bubble], 0, (0, 255, 0), 2) # 緑色
        cv2.drawMarker(img_result, center_of_mass, (0, 0, 255), cv2.MARKER_CROSS, markerSize=10, thickness=2) # 赤色

        # 結果を保存 - 同様にエンコーディング問題に対処
        try:
            # Windowsのファイルパスの問題を回避するため、imencode/imwriteを使用
            _, buf = cv2.imencode('.bmp', img_result)
            buf.tofile(output_path)
        except Exception as e:
            print(f"結果の保存に失敗しました: {output_path} - {str(e)}")
        
        return {
            'volume': volume_mm3,
            'radius': equivalent_radius_mm,
            'center': (center_of_mass[0]/calibration, center_of_mass[1]/calibration)
        }
    else:
        # 気泡が検出されない場合でも、壁面が描画された画像を保存
        try:
            _, buf = cv2.imencode('.bmp', img_result)
            buf.tofile(output_path)
        except Exception as e:
            print(f"結果の保存に失敗しました: {output_path} - {str(e)}")
        print(f"気泡を検出できませんでした: {image_path}")
        return None
    
def process_all_folders(start_folder, end_folder, base_path, calibration, time_interval, start_image_num):
    # 基準値ファイルのパスを正規化 - バックスラッシュを使用
    reference_path = os.path.join(base_path, 'reference.xlsx').replace('/', '\\')
    
    # 基準値ファイルの存在確認
    if not os.path.isfile(reference_path):
        print(f"基準値ファイルが存在しません: {reference_path}")
        return
        
    # 基準値の読み込み
    try:
        reference_df = pd.read_excel(reference_path)
        # reference_dictには、フォルダ番号をキーとして'base_x(pix)'の値を格納
        reference_dict = dict(zip(reference_df['Folder'], reference_df['base_x(pix)']))
        print("基準値の読み込みが完了しました")
    except Exception as e:
        print(f"基準値ファイルの読み込みに失敗しました: {str(e)}")
        return

    # 結果保存用のフォルダを作成 - バックスラッシュを使用
    results_base_path = os.path.join(base_path, "results").replace('/', '\\')
    os.makedirs(results_base_path, exist_ok=True)
    
    # Excelファイルの作成準備
    excel_path = os.path.join(results_base_path, 'analysis_results.xlsx').replace('/', '\\')
    
    # データを格納するための辞書
    all_data = {}
    max_radii = {}    # 各フォルダの最大半径を保存
    gammas = {}       # 各フォルダのγ値を保存

    # 各フォルダを処理
    for folder_num in range(start_folder, end_folder + 1):
        folder_name = str(folder_num)
        folder_path = os.path.join(base_path, folder_name).replace('/', '\\')
        
        if not os.path.exists(folder_path):
            print(f"フォルダが存在しません: {folder_path}")
            continue
            
        # 結果保存用のフォルダを作成
        result_folder = os.path.join(results_base_path, folder_name).replace('/', '\\')
        os.makedirs(result_folder, exist_ok=True)
        
        print(f"フォルダ {folder_name} の処理を開始...")
        
        folder_data = []
        max_radius = 0

        # まず半径データを収集（t*計算のために最大半径が必要なため）
        radius_data = []
        # 現在のフォルダの壁面X座標を取得
        current_wall_x_pixel = reference_dict.get(folder_num, None)

        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith('.bmp'):
                image_path = os.path.join(folder_path, file_name).replace('/', '\\')
                output_path = os.path.join(result_folder, f"{os.path.splitext(file_name)[0]}_result.bmp").replace('/', '\\')
                
                try:
                    # ここでは壁面描画のためだけに calculate_bubble_properties を呼び出す (結果は一旦無視)
                    # 壁面情報も渡すように変更
                    result = calculate_bubble_properties(image_path, output_path, calibration, wall_x_pixel=current_wall_x_pixel)
                    if result:
                        radius_data.append(result['radius'])
                    else:
                        radius_data.append(0) # 結果が取得できない場合は0を追加
                except Exception as e:
                    print(f"エラーが発生しました - {file_name}: {str(e)}")
                    radius_data.append(0) # エラー時も0を追加してインデックスの整合性を保つ

        # データがない場合は次のフォルダへ
        if not radius_data:
            print(f"フォルダ {folder_name} にデータがありません")
            continue
            
        # 最大半径と崩壊点のインデックスを取得
        max_index, collapse_index = find_bubble_points(radius_data)

        # t*計算のための最大半径を取得 (0でなければ)
        rmax_for_t_star = 0
        if max_index != -1 and max_index < len(radius_data):
            rmax_for_t_star = radius_data[max_index]

        # 実際のデータ収集とExcelデータ生成のためのループ
        for idx, file_name in enumerate(sorted(os.listdir(folder_path))):
            if file_name.endswith('.bmp'):
                image_path = os.path.join(folder_path, file_name).replace('/', '\\')
                output_path = os.path.join(result_folder, f"{os.path.splitext(file_name)[0]}_result.bmp").replace('/', '\\')
                
                try:
                    # ここで再度 calculate_bubble_properties を呼び出して実際のデータと描画を行う
                    # 壁面情報も渡すように変更
                    result = calculate_bubble_properties(image_path, output_path, calibration, wall_x_pixel=current_wall_x_pixel)
                    if result:
                        # t*の計算
                        if idx < start_image_num - 1 or rmax_for_t_star == 0:
                            t_star = 0
                        else:
                            elapsed_time = (idx - (start_image_num - 1)) * time_interval
                            t_star = calculate_t_star(elapsed_time, rmax_for_t_star)
                        
                        folder_data.append({
                            't_star': t_star,
                            'volume': result['volume'],
                            'radius': result['radius'],
                            'center_x': result['center'][0],
                            'center_y': result['center'][1]
                        })
                        max_radius = max(max_radius, result['radius'])
                except Exception as e:
                    print(f"エラーが発生しました - {file_name}: {str(e)}")
                    # エラー時はデータを追加しない (スキップ)
        
        # データが少なすぎる場合は次のフォルダへ
        if len(folder_data) < 5:
            print(f"フォルダ {folder_name} の有効なデータが不足しています")
            continue
            
        # 120フレームまでのデータを保存
        all_data[folder_name] = folder_data[:120]
        max_radii[folder_name] = max_radius

        # γの計算
        if len(folder_data) >= 8 and folder_num in reference_dict:
            base_x = reference_dict[folder_num]
            # folder_dataのインデックスが存在するか確認
            if len(folder_data) > 10: # 11番目のデータ (インデックス10) が存在するか確認
                initial_x = folder_data[10]['center_x']
                # 分母が0になるのを防止
                if max_radius > 0:
                    gamma = (initial_x - base_x/calibration) / max_radius
                else:
                    gamma = 0
                gammas[folder_name] = gamma
            else:
                gammas[folder_name] = 0 # データ不足の場合は0とする


    # 処理したフォルダがない場合
    if not all_data:
        print("処理できるデータがありませんでした")
        return
        
    # Excelファイルのデータ作成
    num_folders = len(all_data)
    num_cols = 1 + num_folders * 5
    # ヘッダー行と情報行の分も考慮して行数を調整
    df = pd.DataFrame(np.nan, index=range(120 + 5), columns=range(num_cols)) # データ120行 + ヘッダー5行

    # 最初の列にFolder, Rmax, γを設定
    df.iloc[1, 0] = 'Folder'
    df.iloc[2, 0] = 'Rmax'
    df.iloc[3, 0] = 'γ'

    # 各フォルダのデータを配置
    folder_idx = 0
    for folder_num in range(start_folder, end_folder + 1):
        folder_name = str(folder_num)
        if folder_name not in all_data:
            continue
            
        col_start = 1 + folder_idx * 5  # 各フォルダの開始列

        df.iloc[1, col_start + 2] = folder_num  # フォルダ番号
        df.iloc[2, col_start + 2] = max_radii.get(folder_name, 0)  # Rmax値
        df.iloc[3, col_start + 2] = gammas.get(folder_name, 0)  # γ値

        # カラム名を設定（4行目、インデックスは3）
        df.iloc[4, col_start:col_start+5] = ['t*', '体積', '半径', '重心X', '重心Y']

        # データを配置（5行目から、インデックスは4から）
        for row_idx, data in enumerate(all_data[folder_name]):
            if row_idx < 120:  # 120行までのデータのみを保存
                data_row = row_idx + 5  # データは6行目から開始 (Excelの行番号) -> DataFrameのインデックスは5から
                df.iloc[data_row, col_start] = data['t_star']
                df.iloc[data_row, col_start+1] = data['volume']
                df.iloc[data_row, col_start+2] = data['radius']
                df.iloc[data_row, col_start+3] = data['center_x']
                df.iloc[data_row, col_start+4] = data['center_y']
                
        folder_idx += 1

    # Excelファイルに保存
    try:
        writer = pd.ExcelWriter(excel_path, engine='openpyxl')
        df.to_excel(writer, index=False, header=False)
        
        # ワークシートを取得
        worksheet = writer.sheets['Sheet1']
        
        # 各フォルダの列に対して色付け
        color_folder_idx = 0
        for folder_num in range(start_folder, end_folder + 1):
            folder_name = str(folder_num)
            if folder_name not in all_data:
                continue
                
            col_start = 1 + color_folder_idx * 5  # 各フォルダの開始列
            radius_data = [data['radius'] for data in all_data[folder_name]]
            max_index, collapse_index = find_bubble_points(radius_data)
            
            # Excelの行番号は1から始まるため +1
            # DataFrameのインデックスは0から始まるため、データ開始行のインデックス5を考慮
            
            # インデックスが範囲内かチェック
            if 0 <= max_index < len(radius_data):
                # 最大半径時のセルを赤色で塗る
                max_row_excel = max_index + 6  # Excelの行番号
                cell = worksheet.cell(row=max_row_excel, column=col_start+2)  # 半径の列
                cell.fill = PatternFill(fgColor='FF0000', fill_type='solid')
            
            # インデックスが範囲内かチェック
            if 0 <= collapse_index < len(radius_data):
                # 崩壊時のセルを黄色で塗る
                collapse_row_excel = collapse_index + 6 # Excelの行番号
                cell = worksheet.cell(row=collapse_row_excel, column=col_start+2)
                cell.fill = PatternFill(fgColor='FFFF00', fill_type='solid')
                
            color_folder_idx += 1

        # Excelファイルを保存
        writer.close()
        print(f"結果は {excel_path} に保存されました")
    except Exception as e:
        print(f"Excelファイルの保存に失敗しました: {str(e)}")

    print("全ての処理が完了しました")

if __name__ == "__main__":
    # 処理の設定 - バックスラッシュを使用
    base_path = r'C:\流体工学研究室\実験データ\20231218'  # raw文字列として指定
    start_folder = 1
    end_folder = 77
    calibration = 38
    time_interval = 10  # 時間間隔s（秒）
    start_image_num = 7  # t*=0とする画像番号

    try:
        process_all_folders(start_folder, end_folder, base_path, calibration, time_interval, start_image_num)
    except Exception as e:
        print(f"プログラム実行中にエラーが発生しました: {str(e)}")