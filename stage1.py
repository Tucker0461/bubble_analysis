import cv2
import numpy as np
import os
import pandas as pd
import math 

def get_bubble_properties_and_draw(binary, img_result_visual, wall_x_pixel):
    """
    2値化画像から輪郭を抽出し、面積でフィルタリング後、
    個々の重心を描画し、最大5個の気泡で全体の重心を計算して画像に描画する。
    """
    height, width = binary.shape[:2]
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wall_x = int(wall_x_pixel) if wall_x_pixel is not None else -1 
    
    # 壁面位置の描画
    if wall_x != -1:
        for y_coord in range(0, height, 10):
            cv2.line(img_result_visual, (wall_x, y_coord), (wall_x, min(y_coord + 5, height)), (0, 0, 255), 2)
            
    # 【変更点1: 面積フィルタリングと格納】
    # (面積, 輪郭)のタプルを格納し、後でソートできるようにする
    filtered_bubbles_with_area = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area_pixel2:
            filtered_bubbles_with_area.append((area, contour))

    # 【変更点2: 面積の降順にソートし、上位5個を選択】
    # 面積が大きい順にソート
    filtered_bubbles_with_area.sort(key=lambda x: x[0], reverse=True)
    
    # 全体の重心計算に使用する輪郭 (最大5個) を抽出
    bubbles_for_total_com = [item[1] for item in filtered_bubbles_with_area[:5]] 
    
    # 【変更点3: 上位5個の気泡で全体のモーメントを計算】
    total_m10 = 0
    total_m01 = 0
    total_m00 = 0

    for bubble in bubbles_for_total_com:
        M = cv2.moments(bubble)
        total_m10 += M["m10"]
        total_m01 += M["m01"]
        total_m00 += M["m00"]

    # 気泡の境界と個々の重心を描画 (これは面積フィルタを通過した全ての気泡に対して実行)
    for area, bubble in filtered_bubbles_with_area:
        
        # 緑色の輪郭を描画 (気泡の境界)
        cv2.drawContours(img_result_visual, [bubble], -1, (0, 255, 0), 2) 

        # 個々の重心計算 (描画用)
        M = cv2.moments(bubble)
        if M["m00"] != 0:
            # 重心座標は整数に変換
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # 赤色の重心を描画
            cv2.circle(img_result_visual, (cx, cy), 5, (0, 0, 255), -1)

    if total_m00 != 0:
        # 全体の重心座標を計算
        center_of_mass_x = int(total_m10 / total_m00)
        center_of_mass_y = int(total_m01 / total_m00)
        
        # 全体の重心を異なる色 (例: シアン: BGRで(255, 255, 0)) で描画
        cv2.drawMarker(img_result_visual, (center_of_mass_x, center_of_mass_y), (255, 255, 0), 0, 10)        

# --- Stage 1 のメインロジック ---

def load_background_image(folder_path, blur_x, blur_y, background_no):
    """
    指定されたフォルダの6番目の画像 (インデックス 5) を読み込み、ブラー処理した背景画像として返す。
    """
    file_list = sorted(os.listdir(folder_path))
    
    background_index = background_no
    if len(file_list) <= background_index:
        print(f"エラー: フォルダ {os.path.basename(folder_path)} には6枚以上の画像がありません。背景差分をスキップします。")
        return None

    bg_file_name = file_list[background_index]
    bg_image_path = os.path.join(folder_path, bg_file_name).replace('/', '\\')
    
    try:
        bg_img = cv2.imdecode(np.fromfile(bg_image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if bg_img is None:
            print(f"エラー: 背景画像 {bg_file_name} の読み込みに失敗しました。")
            return None
        
        blurred_background = cv2.GaussianBlur(bg_img, (blur_x, blur_y), 0)
        return blurred_background
        
    except Exception as e:
        print(f"エラー: 背景画像読み込み時にエラーが発生しました: {str(e)}")
        return None


def process_and_save_binary(image_path, binary_output_path, result_output_path, blurred_background, blur_x, blur_y, morph_x, morph_y, itr, diff_threshold, wall_x_pixel, min_area_pixel2):
    """
    画像を読み込み、背景差分で2値化し、結果を描画して保存する。
    """
    if not os.path.isfile(image_path):
        print(f"画像ファイルが存在しません: {image_path}")
        return False
    
    try:
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"画像の読み込みに失敗しました: {image_path}")
            return False
    except Exception as e:
        print(f"画像読み込み時にエラーが発生しました: {image_path} - {str(e)}")
        return False

    # 1. 画像の前処理と2値化
    blurred_current = cv2.GaussianBlur(img, (blur_x, blur_y), 0)
    
    if blurred_background is None or blurred_current.shape != blurred_background.shape:
        binary = np.zeros_like(img, dtype=np.uint8) 
    else:
        diff = cv2.absdiff(blurred_current, blurred_background)
        _, binary = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)


    # 2. モルフォロジー演算
    kernel = np.ones((morph_x, morph_y), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=itr)

    # ----------------------------------------------------
    # 【追加: 穴埋め（Fill Holes）処理のロジック】
    # ----------------------------------------------------
    # 輪郭検出 (RETR_CCOMPまたはRETR_TREEを使用して階層構造を取得)
    # ここではRETR_CCOMPを使用し、外部輪郭と内部輪郭を識別
    # binary_fill_copyは破壊を防ぐために用意 (findContoursは破壊的ではないが、念のため)
    binary_fill_copy = binary.copy()
    contours_fill, hierarchy = cv2.findContours(binary_fill_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # 内部の輪郭 (穴) を白で塗りつぶす
    if contours_fill and hierarchy is not None:
        for i in range(len(contours_fill)):
            # hierarchy[0][i][3] は親の輪郭のインデックスを示す
            # 親が存在する場合 (つまり、内部の輪郭=穴 の場合)
            if hierarchy[0][i][3] != -1:
                # 穴を白 (255) で塗りつぶす
                # thickness=-1 は内部を塗りつぶすことを意味する
                cv2.drawContours(binary, contours_fill, i, 255, -1) 
    # ----------------------------------------------------

    contours_remove, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 閾値以下の面積を持つ輪郭を黒(0)で塗りつぶす
    for contour in contours_remove:
        area = cv2.contourArea(contour)
        if area < min_area_pixel2:
            # 黒(0)で塗りつぶし、除去する
            cv2.drawContours(binary, [contour], -1, 0, -1)
    
    # 3. 可視化（描画と保存）
    # 可視化用の画像を用意 (カラー画像)
    img_result_visual = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # 輪郭検出、フィルタリング、重心計算、描画を実行
    get_bubble_properties_and_draw(binary, img_result_visual, wall_x_pixel)

    # 結果画像 (境界、重心、壁面を描画) を保存
    try:
        _, buf = cv2.imencode('.bmp', img_result_visual)
        buf.tofile(result_output_path)
    except Exception as e:
        print(f"結果の保存に失敗しました: {result_output_path} - {str(e)}")

    # 4. 中間ファイルとして2値化画像を保存
    try:
        _, buf_binary = cv2.imencode('.bmp', binary)
        buf_binary.tofile(binary_output_path)
        return True
    except Exception as e:
        print(f"2値化画像の保存に失敗しました: {binary_output_path} - {str(e)}")
        return False


def stage1_main(base_path, start_folder, end_folder, blur_x, blur_y, morph_x, morph_y, itr, diff_threshold, min_area_pixel2):
    # 基準値ファイルのパスを正規化
    reference_path = os.path.join(base_path, 'reference.xlsx').replace('/', '\\')
    
    if not os.path.isfile(reference_path):
        print(f"基準値ファイルが存在しません: {reference_path}")
        return
        
    try:
        reference_df = pd.read_excel(reference_path)
        reference_dict = dict(zip(reference_df['Folder'], reference_df['base_x(pix)']))
        print("Stage 1: 基準値の読み込みが完了しました")
    except Exception as e:
        print(f"Stage 1: 基準値ファイルの読み込みに失敗しました: {str(e)}")
        return

    binary_base_path = os.path.join(base_path, "binary_images").replace('/', '\\')
    results_base_path = os.path.join(base_path, "results").replace('/', '\\')
    os.makedirs(binary_base_path, exist_ok=True)
    os.makedirs(results_base_path, exist_ok=True)
    
    print("\n--- Stage 1: 2値化、描画、保存を開始 ---")
    
    for folder_num in range(start_folder, end_folder + 1):
        folder_name = str(folder_num)
        folder_path = os.path.join(base_path, folder_name).replace('/', '\\')
        
        if not os.path.exists(folder_path):
            continue
            
        binary_folder = os.path.join(binary_base_path, folder_name).replace('/', '\\')
        result_folder = os.path.join(results_base_path, folder_name).replace('/', '\\')
        os.makedirs(binary_folder, exist_ok=True)
        os.makedirs(result_folder, exist_ok=True) 

        current_wall_x_pixel = reference_dict.get(folder_num, None)
        
        print(f"フォルダ {folder_name} の処理中...")
        
        blurred_background = load_background_image(folder_path, blur_x, blur_y, background_no)

        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith('.bmp'):
                image_path = os.path.join(folder_path, file_name).replace('/', '\\')
                binary_output_path = os.path.join(binary_folder, f"{os.path.splitext(file_name)[0]}_binary.bmp").replace('/', '\\')
                # 【修正点】ファイル名指定: _result.bmp
                result_output_path = os.path.join(result_folder, f"{os.path.splitext(file_name)[0]}_result.bmp").replace('/', '\\')
                
                process_and_save_binary(image_path, binary_output_path, result_output_path, blurred_background, blur_x, blur_y, morph_x, morph_y, itr, diff_threshold, wall_x_pixel=current_wall_x_pixel, min_area_pixel2=min_area_pixel2)
                
    print("Stage 1 完了: 全ての2値化画像と結果描画画像を保存しました。")

if __name__ == "__main__":
    # --- Stage 1 設定 ---
    base_path = r'C:\Research\exp_data\20250611' 
    start_folder = 63
    end_folder = 63
    background_no = 120

    # 2値化パラメータ
    diff_threshold = 10 # 背景差分後の閾値
    blur_x = blur_y = 1
    morph_x = morph_y = 3
    itr = 2
    min_area_pixel2 = 0

    try:
        stage1_main(base_path, start_folder, end_folder, blur_x, blur_y, morph_x, morph_y, itr, diff_threshold, min_area_pixel2)
    except Exception as e:
        print(f"Stage 1 実行中にエラーが発生しました: {str(e)}")