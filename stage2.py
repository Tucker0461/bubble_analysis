import cv2
import numpy as np
import math
import os
import pandas as pd
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import PatternFill
import sys 

# T*計算関数
def calculate_t_star(time, rmax):
    """
    t*を計算する関数
    """
    rho = 1000 # 水の密度 (kg/m³)
    delta_p = 1e5 # 圧力差 (Pa)
    
    if rmax == 0:
        return 0
        
    denominator = 0.91468 * (rmax / 1000) * (rho / delta_p)**0.5
    return time / denominator

# 最大・崩壊点検出関数
def find_bubble_points(radius_data):
    """
    気泡の最大半径点と最初の極小点（崩壊点）を見つける
    """
    max_radius = 0
    max_index = -1
    collapse_index = -1

    started = False
    for i, radius in enumerate(radius_data):
        if not started and radius > 0:
            started = True
        if started:
            if radius > max_radius:
                max_radius = radius
                max_index = i

    radius_threshold = 1.00
    if max_index != -1:
        for i in range(max_index + 1, len(radius_data)):
            if i > 0 and i < len(radius_data) - 1:
                if radius_data[i] < radius_threshold and radius_data[i] < radius_data[i-1] and radius_data[i] <= radius_data[i+1]:
                    collapse_index = i
                    break 
            elif i == len(radius_data) - 1:
                if radius_data[i] < radius_data[i-1]:
                    collapse_index = i
                    break
    
    if collapse_index == -1:
        for i in range(max_index + 1, len(radius_data)):
            if radius_data[i] == 0:
                collapse_index = i
                break
    
    return max_index, collapse_index

# 計算関数
def calculate_properties_from_binary(binary_path, wall_x_pixel, calibration, min_area_pixel2, max_individual_bubbles=4):
    
    empty_agg = {k: 0 for k in ['volume', 'radius', 'center_x', 'center_y', 'x_min_mm', 'x_max_mm', 'y_min_mm', 'y_max_mm', 'v_agarose', 'v_water', 'x_agarose', 'y_agarose', 'x_water', 'y_water', 'aspect_ratio']}

    if not os.path.isfile(binary_path):
        return empty_agg, []
    
    try:
        binary = cv2.imdecode(np.fromfile(binary_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if binary is None:
            return empty_agg, []
            
        wall_x = int(wall_x_pixel) if wall_x_pixel is not None else -1 
        height, width = binary.shape[:2]

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_bubble_data = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area_pixel2:
                
                x, y, w, h = cv2.boundingRect(contour)
                mask = np.zeros(binary.shape, dtype=np.uint8)
                cv2.drawContours(mask, [contour], 0, 255, -1)
                
                volume_total_pixel3 = 0
                moment_x_total = 0
                moment_y_total = 0
                volume_agarose_sum = 0
                moment_x_agarose = 0
                moment_y_agarose = 0
                volume_water_sum = 0
                moment_x_water = 0
                moment_y_water = 0
                
                for j in range(w):
                    x_abs = x + j
                    slice_mask = mask[y:y+h, x_abs] 
                    
                    if np.sum(slice_mask) > 0:
                        non_zero_y = np.nonzero(slice_mask)[0]
                        y_top_rel = non_zero_y[0]
                        y_bottom_rel = non_zero_y[-1]
                        
                        slice_height = y_bottom_rel - y_top_rel + 1
                        r_pixel = slice_height / 2
                        y_center_abs = y + y_top_rel + r_pixel - 0.5

                        v_slice_pixel3 = math.pi * r_pixel**2 * 1 

                        volume_total_pixel3 += v_slice_pixel3
                        moment_x_total += x_abs * v_slice_pixel3
                        moment_y_total += y_center_abs * v_slice_pixel3

                        if wall_x != -1:
                            if x_abs < wall_x: 
                                volume_agarose_sum += v_slice_pixel3
                                moment_x_agarose += x_abs * v_slice_pixel3
                                moment_y_agarose += y_center_abs * v_slice_pixel3
                            elif x_abs >= wall_x: 
                                volume_water_sum += v_slice_pixel3
                                moment_x_water += x_abs * v_slice_pixel3
                                moment_y_water += y_center_abs * v_slice_pixel3
                        else:
                            volume_agarose_sum += v_slice_pixel3 
                            moment_x_agarose += x_abs * v_slice_pixel3
                            moment_y_agarose += y_center_abs * v_slice_pixel3

                
                if volume_total_pixel3 > 0:
                    center_x_pixel = moment_x_total / volume_total_pixel3
                    center_y_pixel = moment_y_total / volume_total_pixel3
                    aspect_ratio = h / w if w != 0 else 0
                else:
                    center_x_pixel, center_y_pixel, aspect_ratio = 0, 0, 0

                all_bubble_data.append({
                    'volume_pixel3': volume_total_pixel3,
                    'center_x_pixel': center_x_pixel,
                    'center_y_pixel': center_y_pixel,
                    'aspect_ratio': aspect_ratio,
                    'min_x_pix': x, 'max_x_pix': x + w, 
                    'min_y_pix': y, 'max_y_pix': y + h,
                    'v_agarose_pix3': volume_agarose_sum,
                    'm_x_agarose': moment_x_agarose,
                    'm_y_agarose': moment_y_agarose,
                    'v_water_pix3': volume_water_sum,
                    'm_x_water': moment_x_water,
                    'm_y_water': moment_y_water,
                })

        # ----------------------------------------------------
        # 2. フレーム集計データ (シート1用)
        # ----------------------------------------------------
        
        total_volume_pixel3 = sum(d['volume_pixel3'] for d in all_bubble_data)
        
        total_moment_x = sum(d['volume_pixel3'] * d['center_x_pixel'] for d in all_bubble_data)
        total_moment_y = sum(d['volume_pixel3'] * d['center_y_pixel'] for d in all_bubble_data)
        
        v_agarose_sum_pix3 = sum(d['v_agarose_pix3'] for d in all_bubble_data)
        m_x_agarose_sum = sum(d['m_x_agarose'] for d in all_bubble_data)
        m_y_agarose_sum = sum(d['m_y_agarose'] for d in all_bubble_data)
        
        v_water_sum_pix3 = sum(d['v_water_pix3'] for d in all_bubble_data)
        m_x_water_sum = sum(d['m_x_water'] for d in all_bubble_data)
        m_y_water_sum = sum(d['m_y_water'] for d in all_bubble_data)
        
        global_min_x = min(d['min_x_pix'] for d in all_bubble_data) if all_bubble_data else 0
        global_max_x = max(d['max_x_pix'] for d in all_bubble_data) if all_bubble_data else 0
        global_min_y = min(d['min_y_pix'] for d in all_bubble_data) if all_bubble_data else 0
        global_max_y = max(d['max_y_pix'] for d in all_bubble_data) if all_bubble_data else 0
        
        if total_volume_pixel3 > 0:
            volume_total_mm3 = total_volume_pixel3 / (calibration**3)
            
            center_x_mm = (total_moment_x / total_volume_pixel3) / calibration
            center_y_mm = (total_moment_y / total_volume_pixel3) / calibration
            
            equivalent_radius_mm = (3 * volume_total_mm3 / (4 * math.pi))**(1/3) 
            
            v_agarose_mm3 = v_agarose_sum_pix3 / (calibration**3)
            v_water_mm3 = v_water_sum_pix3 / (calibration**3)
            
            x_agarose_mm = (m_x_agarose_sum / v_agarose_sum_pix3) / calibration if v_agarose_sum_pix3 > 0 else 0
            y_agarose_mm = (m_y_agarose_sum / v_agarose_sum_pix3) / calibration if v_agarose_sum_pix3 > 0 else 0
            
            x_water_mm = (m_x_water_sum / v_water_sum_pix3) / calibration if v_water_sum_pix3 > 0 else 0
            y_water_mm = (m_y_water_sum / v_water_sum_pix3) / calibration if v_water_sum_pix3 > 0 else 0
            
            max_bubble = max(all_bubble_data, key=lambda x: x['volume_pixel3'])
            max_bubble_ar = max_bubble['aspect_ratio']
            
        else:
            volume_total_mm3 = equivalent_radius_mm = center_x_mm = center_y_mm = 0
            v_agarose_mm3 = v_water_mm3 = x_agarose_mm = y_agarose_mm = x_water_mm = y_water_mm = max_bubble_ar = 0
            
        aggregate_data = {
            'volume': volume_total_mm3,
            'radius': equivalent_radius_mm,
            'center_x': center_x_mm,
            'center_y': center_y_mm,
            'x_min_mm': global_min_x / calibration,
            'x_max_mm': global_max_x / calibration,
            'y_min_mm': global_min_y / calibration,
            'y_max_mm': global_max_y / calibration,
            'v_agarose': v_agarose_mm3,
            'v_water': v_water_mm3,
            'x_agarose': x_agarose_mm,
            'y_agarose': y_agarose_mm,
            'x_water': x_water_mm,
            'y_water': y_water_mm,
            'aspect_ratio': max_bubble_ar,
        }

        # ----------------------------------------------------
        # 3. 個別気泡データ (シート2用)
        # ----------------------------------------------------
        
        all_bubble_data.sort(key=lambda x: x['volume_pixel3'], reverse=True)
        
        individual_bubble_data = []
        
        for d in all_bubble_data[:max_individual_bubbles]:
            individual_bubble_data.append({
                'volume': d['volume_pixel3'] / (calibration**3),
                'center_x': d['center_x_pixel'] / calibration,
                'center_y': d['center_y_pixel'] / calibration,
            })
            
        return aggregate_data, individual_bubble_data
    
    except Exception as e:
        print(f"体積計算でエラーが発生しました: {binary_path} - {str(e)}")
        return empty_agg, []


def stage2_main(base_path, start_folder, end_folder, calibration, time_interval, start_image_num, min_area_pixel2, max_individual_bubbles=4, excel_file_name='analysis_result.xlsx'):
    
    # 基準値の読み込み (変更なし)
    reference_path = os.path.join(base_path, 'reference.xlsx').replace('/', '\\')
    if not os.path.isfile(reference_path):
        print(f"基準値ファイルが存在しません: {reference_path}")
        return
        
    try:
        reference_df = pd.read_excel(reference_path)
        reference_dict = dict(zip(reference_df['Folder'], reference_df['base_x(pix)']))
        print("Stage 2: 基準値の読み込みが完了しました")
    except Exception as e:
        print(f"Stage 2: 基準値ファイルの読み込みに失敗しました: {str(e)}")
        return

    results_base_path = os.path.join(base_path, "results").replace('/', '\\')
    binary_base_path = os.path.join(base_path, "binary_images").replace('/', '\\')
    os.makedirs(results_base_path, exist_ok=True)
    
    all_aggregate_data = {}
    all_individual_data = {}
    max_radii = {}
    gammas = {}
    
    excel_path = os.path.join(results_base_path, excel_file_name).replace('/', '\\')

    print("\n--- Stage 2: 体積・重心の計算とExcelへの出力を開始 ---")
    
    folder_list = list(range(start_folder, end_folder + 1))
    
    # ----------------------------------------------------
    # A. t*計算のための Rmax 決定
    # ----------------------------------------------------
    rmax_for_t_star = 0
    all_temp_radius_data = [] 
    
    for folder_num in folder_list:
        folder_name = str(folder_num)
        binary_folder = os.path.join(binary_base_path, folder_name).replace('/', '\\')
        
        if not os.path.exists(binary_folder): continue
        
        current_wall_x_pixel = reference_dict.get(folder_num, None)
        file_list = sorted([f for f in os.listdir(binary_folder) if f.endswith('_binary.bmp')])
        
        temp_radius_data = []
        temp_max_radius_mm = 0
        
        for file_name in file_list:
            binary_path = os.path.join(binary_folder, file_name).replace('/', '\\')
            agg_data, _ = calculate_properties_from_binary(binary_path, current_wall_x_pixel, calibration, min_area_pixel2, 0)
            frame_radius = agg_data.get('radius', 0) if agg_data else 0
            temp_radius_data.append(frame_radius)
            temp_max_radius_mm = max(temp_max_radius_mm, frame_radius)
        
        all_temp_radius_data.extend(temp_radius_data)
        max_radii[folder_name] = temp_max_radius_mm

    max_index, _ = find_bubble_points(all_temp_radius_data)
    rmax_for_t_star = all_temp_radius_data[max_index] if max_index != -1 and max_index < len(all_temp_radius_data) else 0
    
    # ----------------------------------------------------
    # B. 最終データ収集のためのループ
    # ----------------------------------------------------
    
    for folder_num in folder_list:
        folder_name = str(folder_num)
        binary_folder = os.path.join(binary_base_path, folder_name).replace('/', '\\')
        
        if folder_name not in max_radii or not os.path.exists(binary_folder): continue

        print(f"フォルダ {folder_name} のデータ処理中...")
        
        current_wall_x_pixel = reference_dict.get(folder_num, None)
        file_list = sorted([f for f in os.listdir(binary_folder) if f.endswith('_binary.bmp')])
        
        folder_agg_data = []
        folder_indiv_data = []
        
        for idx, file_name in enumerate(file_list):
            binary_path = os.path.join(binary_folder, file_name).replace('/', '\\')
            
            agg_data, indiv_data = calculate_properties_from_binary(binary_path, current_wall_x_pixel, calibration, min_area_pixel2, max_individual_bubbles)

            if idx < start_image_num - 1 or rmax_for_t_star == 0:
                t_star = 0
            else:
                elapsed_time = (idx - (start_image_num - 1)) * time_interval
                t_star = calculate_t_star(elapsed_time, rmax_for_t_star)
            
            if agg_data:
                agg_data['t_star'] = t_star 
                folder_agg_data.append(agg_data) 
            
            folder_indiv_data.append({
                't_star': t_star,
                'bubbles': indiv_data
            })
        
        all_aggregate_data[folder_name] = folder_agg_data
        all_individual_data[folder_name] = folder_indiv_data
        
        # γの計算
        gamma = 0
        if len(folder_agg_data) >= 8 and folder_num in reference_dict and rmax_for_t_star > 0:
            base_x_mm = reference_dict[folder_num] / calibration
            initial_data_index = start_image_num - 1
            
            if initial_data_index < len(folder_indiv_data):
                initial_frame_data = folder_indiv_data[initial_data_index]['bubbles']
                
                if initial_frame_data:
                    initial_x = initial_frame_data[0]['center_x'] 
                    gamma = (initial_x - base_x_mm) / rmax_for_t_star
            
            gammas[folder_name] = gamma
        else:
            gammas[folder_name] = 0

    # ----------------------------------------------------
    # C. Excel出力処理 (openpyxl ネイティブ書き込み)
    # ----------------------------------------------------
    
    agg_cols_data = ['t_star', 'volume', 'radius', 'center_x', 'center_y', 'x_min_mm', 'x_max_mm', 'y_min_mm', 'y_max_mm', 'v_agarose', 'v_water', 'x_agarose', 'y_agarose', 'x_water', 'y_water', 'aspect_ratio']
    indiv_cols_data = ['t_star', 'volume', 'center_x', 'center_y']
    num_indiv_cols_per_bubble = 3 
    
    # 体積列のインデックスを取得 (リスト内の位置は0から数えて 1 )
    VOLUME_DATA_INDEX = agg_cols_data.index('volume') 
    
    # スタイルを定義
    RED_FILL = PatternFill(fgColor='FF0000', fill_type='solid')
    YELLOW_FILL = PatternFill(fgColor='FFFF00', fill_type='solid')
    
    try:
        wb = Workbook()
        ws1 = wb.active
        ws1.title = 'Sheet1_Aggregate'
        ws2 = wb.create_sheet(title='Sheet2_Individual')
        
        current_col = 1
        current_col_s2 = 1
        
        for folder_num in folder_list:
            folder_name = str(folder_num)
            if folder_name not in all_aggregate_data: 
                # 処理をスキップしたフォルダの列をスキップするロジックは不要 (ヘッダーなしでデータ開始)
                continue 
                
            agg_data = all_aggregate_data[folder_name]
            indiv_data = all_individual_data[folder_name]

            # ---------------------
            # Sheet 1: データ書き込みと色付け
            # ---------------------
            
            # 1. メタデータ (行1-4)
            ws1.cell(row=1, column=current_col + 1, value='Folder')
            ws1.cell(row=2, column=current_col + 1, value='Rmax')
            ws1.cell(row=3, column=current_col + 1, value='γ')
            
            ws1.cell(row=1, column=current_col + 2, value=folder_num)
            ws1.cell(row=2, column=current_col + 2, value=max_radii.get(folder_name, 0))
            ws1.cell(row=3, column=current_col + 2, value=gammas.get(folder_name, 0))
            
            # 2. ヘッダー (行5)
            for col_idx, col_name in enumerate(agg_cols_data):
                ws1.cell(row=5, column=current_col + col_idx, value=col_name)

            # 3. データ本体 (行6以降)
            radius_data = [d.get('radius', 0) for d in agg_data]
            max_index, collapse_index = find_bubble_points(radius_data)
            
            # 色付け対象の列インデックス (Excel 1-based)
            volume_col_excel = current_col + VOLUME_DATA_INDEX
            
            for row_idx in range(120):
                data_row = row_idx + 6 # Excelの行番号
                
                if row_idx < len(agg_data):
                    frame_data = agg_data[row_idx]
                    
                    # 4. データ書き込み
                    for col_idx, col_name in enumerate(agg_cols_data):
                        value = frame_data.get(col_name)
                        ws1.cell(row=data_row, column=current_col + col_idx, value=value)
                        
                        # --- 色付けロジック ---
                        # 体積列にのみ適用
                        if col_idx == VOLUME_DATA_INDEX:
                            if row_idx == max_index:
                                ws1.cell(row=data_row, column=current_col + col_idx, value=value).fill = RED_FILL
                            elif row_idx == collapse_index:
                                ws1.cell(row=data_row, column=current_col + col_idx, value=value).fill = YELLOW_FILL
            
            # 次のフォルダの開始列へ
            current_col += len(agg_cols_data)

            # ---------------------
            # Sheet 2: データ書き込み
            # ---------------------
            
            # 1. 1行目にデータ番号 (Folder) を挿入
            ws2.cell(row=1, column=current_col_s2).value = folder_num

            # 2. ヘッダー (行4, 5)
            ws2.cell(row=4, column=current_col_s2, value='t_star') # t*列

            for b_idx in range(max_individual_bubbles):
                col_offset = b_idx * num_indiv_cols_per_bubble
                
                # 4行目: Bubble ID (t*列の次)
                ws2.cell(row=4, column=current_col_s2 + 1 + col_offset, value=f'Bubble {b_idx+1}')
                
                # 5行目: Data Items
                for item_idx, item_name in enumerate(['volume', 'center_x', 'center_y']):
                    ws2.cell(row=5, column=current_col_s2 + 1 + col_offset + item_idx, value=item_name)
            
            # 3. データ本体 (行6以降)
            for row_idx in range(120):
                data_row = row_idx + 6 
                
                if row_idx < len(indiv_data):
                    frame_data = indiv_data[row_idx]
                    
                    # t* 列 (current_col_s2)
                    ws2.cell(row=data_row, column=current_col_s2, value=frame_data['t_star'])
                    
                    # 個別気泡データ
                    for b_idx in range(max_individual_bubbles):
                        if b_idx < len(frame_data['bubbles']):
                            bubble = frame_data['bubbles'][b_idx]
                            col_offset = b_idx * num_indiv_cols_per_bubble
                            
                            ws2.cell(row=data_row, column=current_col_s2 + 1 + col_offset, value=bubble['volume'])
                            ws2.cell(row=data_row, column=current_col_s2 + 1 + col_offset + 1, value=bubble['center_x'])
                            ws2.cell(row=data_row, column=current_col_s2 + 1 + col_offset + 2, value=bubble['center_y'])

            # Sheet 2の次のフォルダ列へ (t*列 + Bubbleデータ列)
            current_col_s2 += 1 + max_individual_bubbles * num_indiv_cols_per_bubble

        # Workbookを保存
        wb.save(excel_path)
        
        print(f"結果は {excel_path} に保存されました。")
    except Exception as e:
        print(f"致命的なエラー: Excelファイルの保存に失敗しました。パスを確認してください: {excel_path}")
        print(f"エラー詳細: {str(e)}")
        sys.exit(1)

    print("Stage 2 完了: 全ての処理が完了しました")

if __name__ == "__main__":
    # --- Stage 2 設定 ---
    base_path = r'C:\Research\exp_data\20250611' 
    start_folder = 2
    end_folder = 116
    
    # 計算パラメータ
    calibration = 39.4
    time_interval = 0.000005
    start_image_num = 7
    min_area_pixel2 = 50 
    max_individual_bubbles = 4

    try:
        stage2_main(base_path, start_folder, end_folder, calibration, time_interval, start_image_num,
                    min_area_pixel2, max_individual_bubbles, excel_file_name="final_analysis_multi_sheet.xlsx")
    except Exception as e:
        print(f"プログラム実行中にエラーが発生しました: {str(e)}")

            #calibration一覧
    #20231210(0.7) - 38
    #20250417(0.4) - 32.2
    #20250611(0.5) - 39.4
    #20250819(0.3) - 39.5