import cv2
import numpy as np
import os
import pandas as pd
import math 

# --- Stage 2 設定に必要な定数 (Stage 1 の if __name__ == "__main__": から抽出) ---
# 【注意】この値は get_bubble_properties_and_draw 関数内でグローバル変数として参照されます。
min_area_pixel2 = 0

def get_bubble_properties_and_draw(binary, img_result_visual, wall_x_pixel):
    """
    2値化画像から輪郭を抽出し、面積でフィルタリング後、
    個々の重心を描画し、最大5個の気泡で全体の重心を計算して画像に描画する。
    (min_area_pixel2 はグローバル変数として参照)
    """
    height, width = binary.shape[:2]
    
    # 輪郭検出
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    wall_x = int(wall_x_pixel) if wall_x_pixel is not None else -1 
    
    # 壁面位置の描画
    if wall_x != -1:
        for y_coord in range(0, height, 10):
            cv2.line(img_result_visual, (wall_x, y_coord), (wall_x, min(y_coord + 5, height)), (0, 0, 255), 2)
            
    # 【面積フィルタリングと格納】
    filtered_bubbles_with_area = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area_pixel2: # <-- グローバルな min_area_pixel2 を参照
            filtered_bubbles_with_area.append((area, contour))

    # 【面積の降順にソートし、上位5個を選択】
    filtered_bubbles_with_area.sort(key=lambda x: x[0], reverse=True)
    
    # 全体の重心計算に使用する輪郭 (最大5個) を抽出
    bubbles_for_total_com = [item[1] for item in filtered_bubbles_with_area[:5]] 
    
    # 【上位5個の気泡で全体のモーメントを計算】
    total_m10 = 0
    total_m01 = 0
    total_m00 = 0

    for bubble in bubbles_for_total_com:
        M = cv2.moments(bubble)
        total_m10 += M["m10"]
        total_m01 += M["m01"]
        total_m00 += M["m00"]

    # 気泡の境界と個々の重心を描画
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
        
        # 全体の重心を異なる色 (シアン: BGRで(255, 255, 0)) で描画
        cv2.drawMarker(img_result_visual, (center_of_mass_x, center_of_mass_y), (255, 255, 0), 0, 10) 
        
    return True

# ----------------------------------------------------------------------
# Stage 2 の処理関数: 2値化処理をスキップし、結果描画のみを実行
# ----------------------------------------------------------------------

def process_and_save_result_from_binary(original_image_path, binary_image_path, result_output_path, wall_x_pixel):
    """
    元の画像と2値化画像を読み込み、輪郭検出、重心計算、描画を行い、結果を保存する。
    (Stage 1 の process_and_save_binary のロジックを流用し、2値化処理をスキップ)
    """
    
    if not os.path.isfile(original_image_path):
        print(f"元の画像ファイルが存在しません: {original_image_path}")
        return False
    
    if not os.path.isfile(binary_image_path):
        print(f"2値化画像ファイルが存在しません: {binary_image_path}")
        return False
        
    try:
        # 元のグレースケール画像を読み込み
        img_gray = cv2.imdecode(np.fromfile(original_image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        # 既存の2値化画像を読み込み (この画像が Stage 1 で処理済みのもの)
        binary = cv2.imdecode(np.fromfile(binary_image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        
        if img_gray is None or binary is None:
            print(f"画像の読み込みに失敗しました。")
            return False
            
    except Exception as e:
        print(f"画像読み込み時にエラーが発生しました: {original_image_path} / {binary_image_path} - {str(e)}")
        return False

    # 1. 2値化、モルフォロジー演算、穴埋め、面積フィルタリングによる除去はスキップ。
    #    読み込んだ 'binary' 画像を直接使用する。

    # 2. 可視化（描画と保存）
    # 可視化用の画像を用意 (カラー画像)
    img_result_visual = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    # 輪郭検出、フィルタリング、重心計算、描画を実行 (min_area_pixel2 はグローバル参照)
    get_bubble_properties_and_draw(binary, img_result_visual, wall_x_pixel)

    # 結果画像 (境界、重心、壁面を描画) を保存
    try:
        _, buf = cv2.imencode('.bmp', img_result_visual)
        buf.tofile(result_output_path)
        return True
    except Exception as e:
        print(f"結果の保存に失敗しました: {result_output_path} - {str(e)}")
        return False


# ----------------------------------------------------------------------
# Stage 2 のメインロジック (stage1_main の書式に準拠)
# ----------------------------------------------------------------------

def stage2_main(base_path, start_folder, end_folder):
    """
    既存の2値化画像 (binary_images) を使用して、結果描画画像 (results) を更新する。
    (Stage 1 のコード構造を流用)
    """
    
    # 基準値ファイルのパスを正規化
    reference_path = os.path.join(base_path, 'reference.xlsx').replace('/', '\\')
    
    if not os.path.isfile(reference_path):
        print(f"基準値ファイルが存在しません: {reference_path}")
        return
        
    try:
        reference_df = pd.read_excel(reference_path)
        reference_dict = dict(zip(reference_df['Folder'].astype(int), reference_df['base_x(pix)']))
        print("Stage 2: 基準値の読み込みが完了しました")
    except Exception as e:
        print(f"Stage 2: 基準値ファイルの読み込みに失敗しました: {str(e)}")
        return

    binary_base_path = os.path.join(base_path, "binary_images").replace('/', '\\')
    results_base_path = os.path.join(base_path, "results").replace('/', '\\')
    os.makedirs(results_base_path, exist_ok=True)
    
    print("\n--- Stage 2: 既存の2値化画像から結果描画画像の更新を開始 ---")
    
    for folder_num in range(start_folder, end_folder + 1):
        folder_name = str(folder_num)
        
        original_folder_path = os.path.join(base_path, folder_name).replace('/', '\\')
        binary_folder_path = os.path.join(binary_base_path, folder_name).replace('/', '\\')
        result_folder_path = os.path.join(results_base_path, folder_name).replace('/', '\\')
        
        if not os.path.exists(original_folder_path):
            continue
            
        os.makedirs(result_folder_path, exist_ok=True) 

        current_wall_x_pixel = reference_dict.get(folder_num, None)
        
        print(f"フォルダ {folder_name} の処理中...")
        
        # 元画像フォルダ内のファイルを走査
        for file_name in sorted(os.listdir(original_folder_path)):
            if file_name.endswith('.bmp'):
                
                file_base = os.path.splitext(file_name)[0] 
                
                # パスの構築
                original_image_path = os.path.join(original_folder_path, file_name).replace('/', '\\')
                binary_image_path = os.path.join(binary_folder_path, f"{file_base}_binary.bmp").replace('/', '\\')
                result_output_path = os.path.join(result_folder_path, f"{file_base}_result.bmp").replace('/', '\\')
                
                # 処理の実行
                process_and_save_result_from_binary(
                    original_image_path=original_image_path,
                    binary_image_path=binary_image_path,
                    result_output_path=result_output_path,
                    wall_x_pixel=current_wall_x_pixel
                )
                
    print("Stage 2 完了: 全ての結果描画画像を更新しました。")


if __name__ == "__main__":
    # --- Stage 2 実行設定 ---
    # 【注意】ベースパス、フォルダ番号は実行環境に合わせてください
    base_path = r'C:\Research\exp_data\20231210' 
    start_folder = 80
    end_folder = 80
    
    # min_area_pixel2 はグローバル変数として定義済み

    try:
        # Stage 2 メイン関数の呼び出し
        stage2_main(base_path, start_folder, end_folder) 
    except Exception as e:
        print(f"Stage 2 実行中にエラーが発生しました: {str(e)}")