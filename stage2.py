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
    t*を計算する関数 (フォルダごとのRmaxを使用)
    """
    rho = 1000 # 水の密度 (kg/m³)
    delta_p = 1e5 # 圧力差 (Pa)
    
    if rmax == 0:
        return 0
        
    denominator = 0.91468 * (rmax / 1000) * (rho / delta_p)**0.5
    return time / denominator

# 最大・崩壊点検出関数 (修正済み: 8枚目以降に検索を限定し、崩壊点以降を無視)
def find_bubble_points(radius_data):
    """
    気泡の最大半径点と最初の極小点（崩壊点）を見つける
    検索は画像の8番目（インデックス7）から開始し、最初の崩壊点までを有効範囲とする
    """
    max_radius = 0
    max_index = -1
    collapse_index = -1
    
    # Rmax/崩壊点の検索開始インデックスを 7 (8枚目) に設定
    START_INDEX = 10
    
    if len(radius_data) <= START_INDEX:
        return -1, -1 # 検索範囲がない
        
    # ----------------------------------------------
    # 1. 崩壊点の先行検出 (検索範囲の決定)
    # ----------------------------------------------
    # 崩壊点が見つかった場合、そこを検索の終了点とする
    # 崩壊点が見つからない場合は、データリストの最後までを検索対象とする
    end_search_index = len(radius_data)
    radius_threshold = 1.00

    # 8枚目以降で崩壊点を最初に検出
    # i=START_INDEX (7) から len(radius_data)-1 まで
    for i in range(START_INDEX, len(radius_data)):
        
        # 局所的な極小点を見つけるロジック (端の処理を含む)
        is_local_minimum = False
        if i > START_INDEX and i < len(radius_data) - 1:
            if radius_data[i] < radius_threshold and radius_data[i] < radius_data[i-1] and radius_data[i] <= radius_data[i+1]:
                is_local_minimum = True
        elif i == len(radius_data) - 1 and radius_data[i] < radius_data[i-1] and i >= START_INDEX:
            is_local_minimum = True
            
        if is_local_minimum:
            collapse_index = i
            end_search_index = i + 1 # 崩壊点を含めて検索を終了
            break
            
        # 0に戻った点も崩壊点と見なす
        if radius_data[i] == 0 and i >= START_INDEX:
            # 既に極小点として検出されていなければ
            if collapse_index == -1: 
                collapse_index = i
            end_search_index = i + 1
            break
            
    # ----------------------------------------------
    # 2. 決定された範囲 (START_INDEX から end_search_index-1) でRmaxを検出
    # ----------------------------------------------
    
    started = False
    for i in range(START_INDEX, end_search_index):
        radius = radius_data[i]
        
        if not started and radius > 0:
            started = True
        if started:
            if radius > max_radius:
                max_radius = radius
                max_index = i
    
    return max_index, collapse_index

# 体積・重心計算関数 (変更なし)
def calculate_properties_from_binary(binary_path, wall_x_pixel, calibration, min_area_pixel2, max_individual_bubbles=4):
    
    empty_agg = {k: 0 for k in ['volume', 'radius', 'center_x', 'center_y', 'x_min_mm', 'x_max_mm', 'y_min_mm', 'y_max_mm', 'v_agarose', 'v_water', 'x_agarose', 'y_agarose', 'x_water', 'y_water', 'aspect_ratio']}

    if binary_path is None or not os.path.isfile(binary_path):
        return empty_agg, []
    
    try:
        binary = cv2.imdecode(np.fromfile(binary_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if binary is None:
            return empty_agg, []
            
        wall_x = int(wall_x_pixel) if wall_x_pixel is not None else -1 
        
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
            
            # 最大気泡のアスペクト比 (集計データに含めるため、ここでは最大体積の気泡のアスペクト比を代表値とする)
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

# ==============================================================================
# フォルダごとのRmaxを使用するよう修正されたメイン関数
# ==============================================================================

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
    max_radii = {} # 各フォルダの最大半径 Rmax (t*計算用)
    gammas = {}
    
    excel_path = os.path.join(results_base_path, excel_file_name).replace('/', '\\')

    print("\n--- Stage 2: 体積・重心の計算とExcelへの出力を開始 ---")
    
    folder_list = list(range(start_folder, end_folder + 1))
    
    # ----------------------------------------------------
    # A. t*計算のための Rmax 決定（フォルダごと）(修正済み: 崩壊点まででRmaxを決定)
    # ----------------------------------------------------
    
    for folder_num in folder_list:
        folder_name = str(folder_num)
        binary_folder = os.path.join(binary_base_path, folder_name).replace('/', '\\')
        
        if not os.path.exists(binary_folder): 
            max_radii[folder_name] = 0
            continue
        
        current_wall_x_pixel = reference_dict.get(folder_num, None)
        file_list = sorted([f for f in os.listdir(binary_folder) if f.endswith('_binary.bmp')])
        
        # 1. 全フレームの半径データを取得
        radius_data_all = []
        for file_name in file_list:
            binary_path = os.path.join(binary_folder, file_name).replace('/', '\\')
            agg_data, _ = calculate_properties_from_binary(binary_path, current_wall_x_pixel, calibration, min_area_pixel2, 0)
            radius_data_all.append(agg_data.get('radius', 0) if agg_data else 0)
        
        # 2. 8枚目以降、崩壊点までの範囲で最大半径 Rmax を決定
        temp_max_radius_mm = 0
        
        # Rmax/崩壊点のインデックスを検出 (インデックス7/8枚目から検索)
        max_index, collapse_index = find_bubble_points(radius_data_all)
        
        # 検索開始インデックス: 10 (11枚目)
        START_INDEX = 10

        # 検索終了インデックス: 崩壊点が見つかった場合、そのインデックス
        end_index = collapse_index if collapse_index != -1 else len(radius_data_all)
        
        # 11枚目から崩壊点までの範囲で最大半径を計算
        for i in range(START_INDEX, end_index):
            if i < len(radius_data_all):
                temp_max_radius_mm = max(temp_max_radius_mm, radius_data_all[i])
            
        # フォルダごとの Rmax を保存
        max_radii[folder_name] = temp_max_radius_mm

    # ----------------------------------------------------
    # B. 最終データ収集のためのループ (修正済み: 7枚目までの集計値を0に設定)
    # ----------------------------------------------------
    
    for folder_num in folder_list:
        folder_name = str(folder_num)
        binary_folder = os.path.join(binary_base_path, folder_name).replace('/', '\\')
        
        current_rmax_for_t_star = max_radii.get(folder_name, 0) # フォルダRmaxを取得
        
        # フォルダの最大半径 Rmax が 0 の場合、またはフォルダが存在しない場合はスキップ
        if current_rmax_for_t_star == 0 or not os.path.exists(binary_folder): continue

        print(f"フォルダ {folder_name} のデータ処理中...")
        
        current_wall_x_pixel = reference_dict.get(folder_num, None)
        file_list = sorted([f for f in os.listdir(binary_folder) if f.endswith('_binary.bmp')])
        
        folder_agg_data = []
        folder_indiv_data = []
        
        for idx, file_name in enumerate(file_list):
            
            # --- 修正ロジックの適用: 7枚目までの集計値は0 ---
            if idx < start_image_num - 1: # start_image_num=7 の場合、idx=0から5 (1枚目から6枚目)
                # 7枚目まではファイルパスをNoneとし、calculate_properties_from_binaryで全0データを得る
                agg_data, indiv_data = calculate_properties_from_binary(None, None, calibration, min_area_pixel2, 0)
            
            elif idx == start_image_num - 1: # idx=6 (7枚目)
                # 7枚目のみ、集計データは0にするが、個別データも0に
                agg_data, indiv_data = calculate_properties_from_binary(None, None, calibration, min_area_pixel2, 0)

            else: # idx >= start_image_num (8枚目以降)
                # 8枚目以降は通常通り計算
                binary_path = os.path.join(binary_folder, file_name).replace('/', '\\')
                agg_data, indiv_data = calculate_properties_from_binary(binary_path, current_wall_x_pixel, calibration, min_area_pixel2, max_individual_bubbles)
            # --------------------------

            # t* の計算
            if idx < start_image_num - 1: # 7枚目 (インデックス6) まで
                t_star = 0
            else: # 8枚目 (インデックス7) 以降
                elapsed_time = (idx - (start_image_num - 1)) * time_interval
                t_star = calculate_t_star(elapsed_time, current_rmax_for_t_star) 
            
            if agg_data:
                agg_data['t_star'] = t_star 
                folder_agg_data.append(agg_data) 
            
            # 個別気泡データ
            folder_indiv_data.append({
                't_star': t_star,
                'bubbles': indiv_data
            })
        
        all_aggregate_data[folder_name] = folder_agg_data
        all_individual_data[folder_name] = folder_indiv_data
        
        # ------------------------------------------------------------------------
        # γの計算 (変更なし)
        # ------------------------------------------------------------------------
        gamma = 0
        current_rmax = current_rmax_for_t_star # Rmax (分母)
        INITIAL_X_INDEX = 9 # 10枚目

        if len(folder_indiv_data) > INITIAL_X_INDEX and folder_num in reference_dict and current_rmax > 0:
            
            base_x_mm = reference_dict[folder_num] / calibration
            initial_frame_data = folder_indiv_data[INITIAL_X_INDEX]['bubbles']
            
            if initial_frame_data:
                # 最大体積の気泡 (インデックス 0) の中心X座標を取得
                initial_x = initial_frame_data[0]['center_x'] 
                gamma = (initial_x - base_x_mm) / current_rmax 
            
            gammas[folder_name] = gamma
        else:
            gammas[folder_name] = 0
        # ------------------------------------------------------------------------

    # ----------------------------------------------------
    # C. Excel出力処理 (変更なし: find_bubble_points の修正により、色付けも意図した範囲になる)
    # ----------------------------------------------------
    
    agg_cols_data = ['t_star', 'volume', 'radius', 'center_x', 'center_y', 'x_min_mm', 'x_max_mm', 'y_min_mm', 'y_max_mm', 'v_agarose', 'v_water', 'x_agarose', 'y_agarose', 'x_water', 'y_water', 'aspect_ratio']
    num_indiv_cols_per_bubble = 3 
    VOLUME_DATA_INDEX = agg_cols_data.index('volume') 
    
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
            # find_bubble_points は 8枚目以降かつ崩壊点までで検索を行う
            max_index, collapse_index = find_bubble_points(radius_data) 
            
            VOLUME_DATA_INDEX_IN_LOOP = agg_cols_data.index('volume')
            
            for row_idx in range(120):
                data_row = row_idx + 6 # Excelの行番号
                
                if row_idx < len(agg_data):
                    frame_data = agg_data[row_idx]
                    
                    # 4. データ書き込み
                    for col_idx, col_name in enumerate(agg_cols_data):
                        value = frame_data.get(col_name)
                        cell = ws1.cell(row=data_row, column=current_col + col_idx, value=value)
                        
                        # --- 色付けロジック ---
                        # max_index/collapse_index は8枚目以降の有効なインデックスのみを持つ
                        if col_idx == VOLUME_DATA_INDEX_IN_LOOP:
                            if row_idx == max_index:
                                cell.fill = RED_FILL
                            elif row_idx == collapse_index:
                                cell.fill = YELLOW_FILL
            
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
    base_path = r'C:\Research\exp_data\20231210' 
    start_folder = 2
    end_folder = 84

    # 計算パラメータ
    calibration = 38
    time_interval = 0.000005
    start_image_num = 7 # t*=0とする画像番号 (1枚目から7枚目までが集計値0、8枚目以降で計算)
    min_area_pixel2 = 0
    max_individual_bubbles = 4

    try:
        stage2_main(base_path, start_folder, end_folder, calibration, time_interval, start_image_num,
                    min_area_pixel2, max_individual_bubbles, excel_file_name="2_analysis_20231210.xlsx")
    except Exception as e:
        print(f"プログラム実行中にエラーが発生しました: {str(e)}")

    #calibration一覧
    #20231210(0.7) - 38
    #20250417(0.4) - 32.2
    #20250611(0.5) - 39.4
    #20250819(0.5) - 39.5