import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.title("YOLO26 物体検出")

# Session State を初期化し、信頼度しきい値のデフォルト値を設定
if "conf" not in st.session_state:
    st.session_state.conf = 0.25

# スライダーで信頼度しきい値を変更する
st.session_state.conf = st.slider(
    "信頼度しきい値",
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    value=st.session_state.conf
)

# 物体検出用の軽量モデル YOLO26n を取得する
model = YOLO("yolo26n.pt")

# ローカルカメラの起動と静止画の取得
img_file_buffer = st.camera_input("カメラで撮影")

# 画像が撮影された場合のみ処理を実行
if img_file_buffer is not None:
    # 画像データをPillowで開き、NumPy配列に変換する
    image = Image.open(img_file_buffer)
    img_array = np.array(image)

    # スライダーの信頼度しきい値を指定して物体検出を実行する
    results = model(img_array, conf=st.session_state.conf)

    # 検出結果（クラス名とバウンディングボックス）を描画した画像を取得する
    # plot() は BGR 順の NumPy 配列を返す
    annotated_img = results[0].plot()

    # BGR 順の画像を指定して表示する
    st.image(annotated_img, channels="BGR", width="stretch")