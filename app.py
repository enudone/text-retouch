import streamlit as st
import cv2
import numpy as np
import gensim

# サイドバーのUI
uploaded_file = st.sidebar.file_uploader("●画像アップロード", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # アップロードされたファイルからバイトデータを読み込み、
    # それを bytearray に変換し、さらにそのデータを np.uint8 型の NumPy 配列に変換
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # file_bytes のデータを、カラー画像としてデコード
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # RGB 形式に変換
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.sidebar.image(image_rgb, width=100) # アップロード画像のプレビュー

input_text = st.sidebar.text_input("●レタッチに適用する単語を入力")
retouch_button = st.sidebar.button("レタッチする")
# download_button = st.sidebar.button("編集後の画像をダウンロード")
st.sidebar.divider()
st.sidebar.write('このアプリは、東北大学 乾・鈴木研究室が作成した日本語エンティティベクトル（2017年2月1日版）を使用しています。詳細およびダウンロードは、東北大学 乾・鈴木研究室のウェブサイト（ http://www.cl.ecei.tohoku.ac.jp/~m-suzuki/jawiki_vector/ ）で確認ができます。このリソースは CC BY-SA 4.0 の下で提供されています。')

# ページタイトル（アプリ名）
st.title("Text-Retouch")

# アプリ説明
st.info("Text-Retouch は日本語の単語を入力することで、その単語の性質やイメージを基に画像をレタッチするアプリです。例えば、「明るい」「光」といった単語で画像を明るく、「暗い」「影」といった単語で画像を暗くレタッチすることができます。")
st.divider()

# 学習済みの日本語ベクトルデータのロード
# @st.cache_data でキャッシュ化する
@st.cache_data
def load_model():
    return gensim.models.KeyedVectors.load_word2vec_format('entity_vector/entity_vector.model.bin', binary=True)

# モデルを定義
model = load_model()

# ターゲットとなる単語と任意の単語リストの各単語とのコサイン類似度の平均を算出するメソッド
def get_similarity_score(target, word_list):
    total_similarity = 0
    for word in word_list:
        try:
            total_similarity += model.similarity(target, word)
        except KeyError:
            raise KeyError # エラーを呼び出し元に伝達
    return total_similarity / len(word_list)

# 入力単語の明るさイメージを判定するためにコサイン類似度を調べる対象の単語リスト
words_bright = ['明るい', '眩しい', '輝く', '光', '日差し', '白']
words_dark = ['暗い', '薄暗い', '陰る', '影', '暗闇', '黒']

# 明るさ調整用スコアを算出するメソッド
def get_luminance_score(target):
    score_bright = get_similarity_score(target, words_bright)
    score_dark = get_similarity_score(target, words_dark)
    score_total = score_bright - score_dark
    return score_total

# 画像の明るさを調整する関数
# img: レタッチ対象の画像
# lumi_score: 明るさ調整用スコア
def adjust_luminance(img,lumi_score):
    adjust_level = 6 # TODO: 調整レベルを5段階で設定できるようにする（2/4/6/8/10）
    alpha = 1.0 + (lumi_score * adjust_level)  # 明るさの修正値を設定
    # print('alpha: ', alpha)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=0)

# 入力単語の鮮やかさイメージを判定するためにコサイン類似度を調べる対象の単語リスト
words_saturation_high = ['鮮明','鮮やか','きらびやか','華やか','濃い','カラフル','ビビッド']
words_saturation_low = ['淡色', '退色', '地味', '控えめ', '薄い', 'モノクロ', 'ソフト']

# 写真の彩度の調整用のスコアを算出するメソッド
# レタッチに使われる入力変数 target を引数に取る
# 出力されたスコアが正の数の場合は彩度をプラス補正、負の数の場合はマイナス補正をかける
def get_saturation_score(target):
    score_s_high = get_similarity_score(target, words_saturation_high)
    score_s_low = get_similarity_score(target, words_saturation_low)
    score_total = score_s_high - score_s_low
    return score_total

# レタッチボタンが押されたときの処理
if retouch_button and uploaded_file and input_text:
    try:
        # 明るさ調整用スコアを取得
        luminance_score = get_luminance_score(input_text)
        # st.write(f"明るさ調整用スコア: {luminance_score}")

        # レタッチ済み画像を取得
        retouched_image_rgb = adjust_luminance(image_rgb, luminance_score)

        # 元画像とレタッチ済み画像を表示
        col1, col2 = st.columns(2)  # 2つのカラムを作成

        with col1:
            st.write("元画像")
            st.image(image_rgb, use_column_width="auto")
        with col2:
            st.write(f"レタッチ画像（適用した単語「{input_text}」）")
            st.image(retouched_image_rgb, use_column_width="auto")
    except KeyError:
        st.error(f"入力された単語「{input_text}」では画像をレタッチするためのスコアを算出できませんでした。より一般的な単語でお試しください。")


