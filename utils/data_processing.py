import glob
import json
import os

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def get_mask_from_json(json_path, image):
    """JSONファイルからマスクを取得する

    Args:
        json_path: JSONファイルのパス
        image: 画像データ

    Returns:
        mask: マスク
        text: テキスト
        is_sentence: 文章かどうかのフラグ
    """
    # JSONファイルの読み込み
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # マスクの初期化
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # テキストとマスクの処理
    text = ""
    is_sentence = False
    
    # データ形式によって処理を分ける
    if "sentence" in data:
        # 文章ベースのセグメンテーション
        text = data["sentence"]
        is_sentence = True
        
        if "mask" in data:
            # COCO RLEフォーマットのマスク
            mask_rle = data["mask"]
            if isinstance(mask_rle, dict):
                # RLE形式の場合はpycocotoolsでデコード
                mask = mask_utils.decode(mask_rle)
            elif isinstance(mask_rle, list):
                # ポリゴン形式の場合はマスクに変換
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly in mask_rle:
                    # ポリゴンをマスクに変換
                    if len(poly) >= 6:  # 最低3点（x,y座標×3）必要
                        poly_array = np.array(poly).reshape(-1, 2)
                        cv2.fillPoly(mask, [poly_array.astype(np.int32)], 1)
    
    elif "object" in data:
        # 物体ベースのセグメンテーション
        text = data["object"]
        
        if "mask" in data:
            mask_data = data["mask"]
            if isinstance(mask_data, dict):
                # RLE形式
                mask = mask_utils.decode(mask_data)
            elif isinstance(mask_data, list):
                # ポリゴン形式
                h, w = image.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly in mask_data:
                    if len(poly) >= 6:
                        poly_array = np.array(poly).reshape(-1, 2)
                        cv2.fillPoly(mask, [poly_array.astype(np.int32)], 1)
    
    else:
        # その他のフォーマット
        for key in data:
            if isinstance(data[key], dict) and "segmentation" in data[key]:
                # COCO形式の場合
                segmentation = data[key]["segmentation"]
                if isinstance(segmentation, dict):
                    # RLE形式
                    mask = mask_utils.decode(segmentation)
                    text = data[key].get("name", key)
                elif isinstance(segmentation, list):
                    # ポリゴン形式
                    h, w = image.shape[:2]
                    mask = np.zeros((h, w), dtype=np.uint8)
                    for poly in segmentation:
                        if len(poly) >= 6:
                            poly_array = np.array(poly).reshape(-1, 2)
                            cv2.fillPoly(mask, [poly_array.astype(np.int32)], 1)
                    text = data[key].get("name", key)
                break
    
    return mask, text, is_sentence


if __name__ == "__main__":
    data_dir = "./train"
    vis_dir = "./vis"

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    json_path_list = sorted(glob.glob(data_dir + "/*.json"))
    for json_path in json_path_list:
        img_path = json_path.replace(".json", ".jpg")
        img = cv2.imread(img_path)[:, :, ::-1]

        # In generated mask, value 1 denotes valid target region, and value 255 stands for region ignored during evaluaiton.
        mask, comments, is_sentence = get_mask_from_json(json_path, img)

        ## visualization. Green for target, and red for ignore.
        valid_mask = (mask == 1).astype(np.float32)[:, :, None]
        ignore_mask = (mask == 255).astype(np.float32)[:, :, None]
        vis_img = img * (1 - valid_mask) * (1 - ignore_mask) + (
            (np.array([0, 255, 0]) * 0.6 + img * 0.4) * valid_mask
            + (np.array([255, 0, 0]) * 0.6 + img * 0.4) * ignore_mask
        )
        vis_img = np.concatenate([img, vis_img], 1)
        vis_path = os.path.join(
            vis_dir, json_path.split("/")[-1].replace(".json", ".jpg")
        )
        cv2.imwrite(vis_path, vis_img[:, :, ::-1])
        print("Visualization has been saved to: ", vis_path)
