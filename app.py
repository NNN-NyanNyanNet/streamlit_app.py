import streamlit as st
import pandas as pd
import numpy as np
from scipy import interpolate, signal
import matplotlib.pyplot as plt
from scipy import signal

# トップページ
st.title("心拍変動解析")
st.write('卒業論文で用いた心拍変動解析のコードをwebアプリにしてみました!')
st.write('サイドバーより、解析の手法を選択してください！')
st.write('')
st.subheader("＊読みこむExcelデータの設定について＊")
st.image(R"images/エクセルデータについて.png", use_container_width=True)
st.write('※エラーが出る場合、列の名前や、ファイルにシートが複数含まれていないか確認してください。')

# サイドバーで解析方法を選択
selected_option = st.sidebar.selectbox(
    "解析方法を選択してください^._.^",
    ["FFT解析", "ローレンツプロット", "トーンエントロピー"]
)

#FFT
if selected_option == "FFT解析":
        st.divider()  # 水平線を挿入
        st.write("（CURRENT CHOICE）: FFT解析")

        st.header('【FFT解析】')
        st.markdown(
            """
            縦列上にTime[s]とRRIaを含むエクセルファイルをアップロードしてください。
            以下を表示します：
            - RRI変動とFFTのパワースペクトル
            - HF値（0.15~0.40 Hz、副交感神経）
            - LF値（0.04~0.15 Hz、交感神経と副交感神経）
            - HF/LF値（交感神経活動と副交感神経活動のバランス）
            - VLF値（0.003~0.04 Hz、体温・熱産生に関与する交感神経活動指標）
            - 総パワー値（TOTAL, 0.00~0.40 Hz、総自律神経活動指標）
            """
        )
        uploaded_file = st.file_uploader("エクセルファイルをアップロードしてください。", type=["xlsx"])

        if uploaded_file:
            st.write("ファイルがアップロードされました！解析を開始します...")
            # データを読み込み
            df = pd.read_excel(uploaded_file)
            st.write("データプレビュー", df.head())

            try:
                # 必要な列を取得
                timestamps = df["Time[s]"]  # タイムスタンプ
                rri = df["RRIa"]  # 心拍間隔データ

                # NaN値を除外してデータを抽出
                valid_indices = ~np.isnan(rri)
                valid_timestamps = timestamps[valid_indices]
                valid_rri = rri[valid_indices]

                # リサンプリングのための新しいタイムスタンプを作成
                new_timestamps = np.arange(valid_timestamps.iloc[0], valid_timestamps.iloc[-1], 0.5)

                # スプライン3次補間関数を作成
                interpolator = interpolate.interp1d(valid_timestamps, valid_rri, kind='cubic', fill_value="extrapolate")
                resampled_rri = interpolator(new_timestamps)

                # 各valid_rriからvalid_rriの平均値を引いた値を計算
                detrended_rri = resampled_rri - np.mean(resampled_rri)

                # ハイパス
                sampling_rate = 2.0
                b, a = signal.butter(4, 0.04 / (0.5 * sampling_rate), btype='high')
                filtered_rri = signal.filtfilt(b, a, detrended_rri)

                # ローパス
                b, a = signal.butter(4, 0.6 / (0.5 * sampling_rate), btype='low')
                filtered_rri = signal.filtfilt(b, a, filtered_rri)

                # ハミングウィンドウ
                window = signal.hamming(len(filtered_rri))
                filtered_rri_hamming = filtered_rri * window

                # FFT解析
                N = len(filtered_rri_hamming)
                freq = np.fft.fftfreq(N, d=1/sampling_rate)
                F = np.fft.fft(filtered_rri_hamming)
                amp = np.abs(F / (N / 2))

                # グラフ表示
                fig, ax = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'hspace': 0.7})
                ax[0].plot(new_timestamps[:len(filtered_rri)], filtered_rri)
                ax[0].set_title("Filtered RRI")
                ax[0].set_xlabel("Time (s)")
                ax[0].set_ylabel("Amplitude")

                ax[1].plot(freq[1:int(N/2)], amp[1:int(N/2)])
                ax[1].set_title("FFT Power Spectrum")
                ax[1].set_xlabel("Frequency (Hz)")
                ax[1].set_ylabel("Amplitude")

                plt.tight_layout(pad=1.0)
                st.pyplot(fig)

                # LF帯域とHF帯域、VLF帯域とトータルパワーの周波数範囲を定義
                lf_freq_range = (0.04, 0.15)
                hf_freq_range = (0.15, 0.4)
                vlf_freq_range = (0.0033, 0.04)
                total_freq_range = (0, 0.4)

                # LF帯域とHF帯域、VLF帯域とトータルパワーを計算
                lf_power = np.sum(amp[(freq >= lf_freq_range[0]) & (freq <= lf_freq_range[1])])
                hf_power = np.sum(amp[(freq >= hf_freq_range[0]) & (freq <= hf_freq_range[1])])
                vlf_power = np.sum(amp[(freq >= vlf_freq_range[0]) & (freq <= vlf_freq_range[1])])
                total_power = np.sum(amp[(freq >= total_freq_range[0]) & (freq <= total_freq_range[1])])

                # LF/HF比を計算
                lf_hf_ratio = lf_power / hf_power

                st.write(f"LF Power: {lf_power:.2f}")
                st.write(f"HF Power: {hf_power:.2f}")
                st.write(f"LF/HF Ratio: {lf_hf_ratio:.2f}")
                st.write(f"vlf_power: {hf_power:.2f}")
                st.write(f"total_power: {hf_power:.2f}")

            except KeyError as e:
                st.error(f"データに必要な列がありません: {e}")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

#ローレンツプロット
if selected_option == "ローレンツプロット":
        st.divider()  # 水平線を挿入
        st.write("（CURRENT CHOICE）: ローレンツプロット")

        st.header('【ローレンツプロット】')
        st.markdown(
            """
            縦列上にRRIaとRRIbを含むエクセルファイルをアップロードしてください。
            以下を表示します：
            - ローレンツプロットのグラフ
            - 楕円形の長径（L）と短径（T）
            - CSI（心臓交感神経指数）
            - CSV（心臓迷走神経指数）
            """
        )
        uploaded_file = st.file_uploader("エクセルファイルをアップロードしてください。", type=["xlsx"])

        if uploaded_file:
            # データを読み込む
            df = pd.read_excel(uploaded_file)
            st.write("データプレビュー:", df.head())

            try:
                # RRIa と RRIb の列を取得
                rria = df["RRIa"]
                rrib = df["RRIb"]

                # 心拍間隔データを使用してポアンカレプロットを作成
                x = 1000 * rria[:-1]  # x軸はRRIaのデータ（最後のデータを除く）
                y = 1000 * rrib[:-1]  # y軸はRRIbのデータ（最後のデータを除く）

                # グラフ描画
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))

                # ローレンツプロットを描画
                ax[0].scatter(x, y, s=10, alpha=0.5)
                ax[0].set_title("Lorenz Plot")
                ax[0].set_xlabel("RRIa (ms)")
                ax[0].set_ylabel("RRIb (ms)")

                # X軸とY軸の表示範囲を550から1050に設定
                ax[0].set_xlim(580, 1050)
                ax[0].set_ylim(580, 1050)

                # X軸とY軸の目盛り線を50ごとに設定
                ax[0].set_xticks(range(600, 1050, 50))
                ax[0].set_yticks(range(600, 1050, 50))

                # 目盛り線を追加
                ax[0].grid(True, linestyle="--", alpha=0.7)

                # 回転行列を定義
                theta = np.pi / 4  # 45度のラジアン
                rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                            [np.sin(theta), np.cos(theta)]])

                # 心拍データを回転させる
                rotated_data = np.dot(np.column_stack((x, y)), rotation_matrix)

                # 回転後のデータを取得
                rotated_x = rotated_data[:, 0]
                rotated_y = rotated_data[:, 1]

                # 回転後のポアンカレプロットを描画
                ax[1].scatter(rotated_x, rotated_y, s=10, alpha=0.5)
                ax[1].set_title("Rotated Poincaré Plot")
                ax[1].set_xlabel("Rotated RRIa (ms)")
                ax[1].set_ylabel("Rotated RRIb (ms)")
                ax[1].grid(True)

                # レイアウト調整
                plt.tight_layout()

                # Streamlitでグラフを表示
                st.pyplot(fig)

                # 標準偏差の計算
                sd1 = np.std(rotated_y)  # SD1: 回転後のy軸方向の標準偏差
                sd2 = np.std(rotated_x)  # SD2: 回転後のx軸方向の標準偏差
                T = 4 * sd1
                L = 4 * sd2

                # CVIとCSIの計算
                cvi = np.log10(L * T)
                csi = L / T

                # 結果を表示
                st.write(f"L): {L:.2f}")
                st.write(f"T: {T:.2f}")
                st.write(f"CSI: {csi:.2f}")
                st.write(f"CVI: {cvi:.2f}")

            except KeyError as e:
                st.error(f"データに必要な列が見つかりません: {e}")
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

#トーンエントロピー
if selected_option == "トーンエントロピー":
        st.divider()  # 水平線を挿入
        st.write("（CURRENT CHOICE）: トーンエントロピー")

        st.header('【トーンエントロピー】')
        st.markdown(
            """
            縦列上にRRIaを含むエクセルファイルをアップロードしてください。
            以下を表示します：
            - トーン変動とエントロピー変動のグラフ
            - TONE値（心臓の加速と抑制のバランス）
            - ENTROPY値（≒総パワー）
            """
        )
        uploaded_file = st.file_uploader("エクセルファイルをアップロードしてください。", type=["xlsx"])

if uploaded_file:
    # データを読み込む
    df = pd.read_excel(uploaded_file)
    st.write("データプレビュー:", df.head())

    try:
        # RRIa列のデータを取得
        rria_data = df['RRIa'].values

        # トーンの計算
        pi_values = (rria_data[:-1] - rria_data[1:]) / rria_data[:-1] * 100
        tone = np.mean(pi_values)

        # エントロピーの計算
        p_i_values, _ = np.histogram(pi_values, bins='auto', density=True)
        non_zero_p_i_values = p_i_values[p_i_values > 0]
        entropy = -np.sum(non_zero_p_i_values * np.log2(non_zero_p_i_values))

        # 計算結果を表示
        st.write(f"Tone (トーン): {tone:.2f}")
        st.write(f"Entropy (エントロピー): {entropy:.2f}")

        # グラフ描画
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))

        # トーンの変化を表示
        ax[0].plot(pi_values, label='PI Values', color='blue')
        ax[0].set_title('Tone Change')
        ax[0].legend()

        # エントロピーの変化を表示
        ax[1].hist(pi_values, bins='auto', density=True, alpha=0.75, color='blue', edgecolor='black')
        ax[1].set_title('Entropy Change')

        # レイアウト調整
        plt.tight_layout()

        # Streamlitでグラフを表示
        st.pyplot(fig)

    except KeyError as e:
        st.error(f"データに必要な列が見つかりません: {e}")
    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
