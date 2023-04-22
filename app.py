import streamlit as st
import cv2
import numpy as np
import gensim

# サイドバーのUI
uploaded_file = st.sidebar.file_uploader("画像をアップロードしてください", type=["jpg", "png", "jpeg"])
input_text = st.sidebar.text_input("レタッチに適用する単語を入力してください")
retouch_button = st.sidebar.button("レタッチする")
download_button = st.sidebar.button("編集後の画像をダウンロード")
st.sidebar.write('このアプリは、東北大学 乾・鈴木研究室が作成した日本語エンティティベクトル（2017年2月1日版）を使用しています。詳細およびダウンロードは、東北大学 乾・鈴木研究室のウェブサイト（ http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/ ）で確認ができます。このリソースは CC BY-SA 4.0 の下で提供されています。')

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
    st.image(image_rgb, width=400)


# 学習済みの日本語ベクトルデータの読み込み
model = gensim.models.KeyedVectors.load_word2vec_format('entity_vector/entity_vector.model.bin', binary=True)

# 入力単語の明るさイメージを判定するためにコサイン類似度を調べる対象の単語リスト
words_bright = ['明るい', '眩しい', '輝く', '光', '日差し', '白']
words_dark = ['暗い', '薄暗い', '陰る', '影', '暗闇', '黒']

# 明るさ調整用スコアを算出するメソッド
def get_luminance_score(target):
    score_bright = get_similarity_score(target, words_bright)
    score_dark = get_similarity_score(target, words_dark)
    score_total = score_bright - score_dark
    return score_total

# ターゲットとなる単語と任意の単語リストの各単語とのコサイン類似度の平均を算出するメソッド
def get_similarity_score(target, word_list):
    total_similarity = 0
    for word in word_list:
        try:
            total_similarity += model.similarity(target, word)
        except KeyError:
            raise KeyError # エラーを呼び出し元に伝達
    return total_similarity / len(word_list)

# レタッチボタンが押されたときの処理
if retouch_button and uploaded_file and input_text:
    try:
        luminance_score = get_luminance_score(input_text)
        st.write(f"明るさ調整用スコア: {luminance_score}")
    except KeyError:
        st.error(f"入力されたテキスト {input_text} では画像をレタッチするためのスコアを算出できませんでした。より一般的な単語でお試しください。")


