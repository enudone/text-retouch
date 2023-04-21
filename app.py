import streamlit as st
import cv2
import numpy as np

# サイドバーのUI
uploaded_file = st.sidebar.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])
input_text = st.sidebar.text_input("レタッチ用のテキストを入力してください")
retouch_button = st.sidebar.button("レタッチする")
download_button = st.sidebar.button("編集後の画像をダウンロード")

# ページタイトル
st.title("Text-Retouch")

# 編集前写真の表示セクション
st.subheader("アップロード画像")
if uploaded_file is None:
    st.write("No Image")
else:
    # アップロードされたファイルからバイトデータを読み込み、
    # それを bytearray に変換し、さらにそのデータを np.uint8 型の NumPy 配列に変換
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # file_bytes のデータを、カラー画像としてデコード
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # RGB 形式に変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, use_column_width=True)

