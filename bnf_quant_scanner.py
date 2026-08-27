import io
import json
from datetime import datetime
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import requests
import yfinance as yf


class BNFTradingScanner:
    """BNF流・逆張り自動スコアリング＆スクリーニングシステム

    5つの絶対組み込み必須要件（地合い完全連動、業種別動態閾値、個別ナイフ検知、
    出来高セリクラ判定、多角乖離多段判定）を完全実装したQuantスキャナー。
    """

    # ═════════════════════════════════════════════════════════════════════════
    # 【要件2】業種（セクター）別・ボラティリティ動態閾値設定
    # ═════════════════════════════════════════════════════════════════════════
    SECTOR_CONFIG: Dict[str, Dict[str, Any]] = {
        # 1. 大型・バリュー株 (買いシグナル閾値: -10.0% 〜 -15.0%)
        "自動車・輸送機器": {"threshold": -10.0, "category": "大型・バリュー"},
        "機械": {"threshold": -10.0, "category": "大型・バリュー"},
        "銀行・金融業": {"threshold": -10.0, "category": "大型・金融バリュー"},
        "商社・卸売業": {"threshold": -10.0, "category": "大型・バリュー"},
        "鉄鋼・非鉄金属": {"threshold": -10.0, "category": "シクリカル・バリュー"},
        "化学・素材": {"threshold": -10.0, "category": "シクリカル・バリュー"},
        # 2. ハイテク・半導体株 (買いシグナル閾値: -12.0% 〜 -18.0%)
        "半導体・電子部品": {"threshold": -12.0, "category": "ハイテク・半導体"},
        "情報通信・IT": {"threshold": -15.0, "category": "ハイテク・IT"},
        "精密機器": {"threshold": -12.0, "category": "ハイテク・精密"},
        "ゲーム・エンタメ": {"threshold": -12.0, "category": "ハイテク・コンテンツ"},
        # 3. 新興・グロース株 (買いシグナル閾値: -20.0% 〜 -30.0%)
        "サービス・グロース": {"threshold": -20.0, "category": "新興・グロース"},
        "ネットEC・プラットフォーム": {
            "threshold": -22.0,
            "category": "新興・グロース",
        },
        # 4. ディフェンシブ株 (買いシグナル閾値: -7.0% 〜 -10.0%)
        "医薬品": {"threshold": -8.0, "category": "ディフェンシブ"},
        "食料品": {"threshold": -7.0, "category": "ディフェンシブ"},
        "陸運・物流": {"threshold": -8.0, "category": "ディフェンシブ"},
        "電気・ガス業": {"threshold": -7.0, "category": "ディフェンシブ"},
        "小売・消費": {"threshold": -10.0, "category": "消費・ディフェンシブ"},
    }

    # デフォルト閾値（未定義セクター用）
    DEFAULT_THRESHOLD: float = -12.0

    # 代表的プライム主力銘柄リスト（全銘柄スキャン時の動的取得失敗時のフォールバック用）
    SAMPLE_TARGETS: List[Dict[str, str]] = [
        {"code": "6920.T", "name": "レーザーテック", "sector": "半導体・電子部品"},
        {"code": "8035.T", "name": "東京エレクトロン", "sector": "半導体・電子部品"},
        {"code": "6857.T", "name": "アドバンテスト", "sector": "半導体・電子部品"},
        {"code": "9984.T", "name": "ソフトバンクG", "sector": "情報通信・IT"},
        {"code": "6758.T", "name": "ソニーG", "sector": "ゲーム・エンタメ"},
        {
            "code": "4385.T",
            "name": "メルカリ",
            "sector": "ネットEC・プラットフォーム",
        },
        {"code": "9101.T", "name": "日本郵船", "sector": "陸運・物流"},
        {"code": "7974.T", "name": "任天堂", "sector": "ゲーム・エンタメ"},
        {"code": "6981.T", "name": "村田製作所", "sector": "半導体・電子部品"},
        {"code": "6098.T", "name": "リクルートHD", "sector": "サービス・グロース"},
        {"code": "7203.T", "name": "トヨタ自動車", "sector": "自動車・輸送機器"},
        {"code": "9983.T", "name": "ファーストリテイリング", "sector": "小売・消費"},
        {"code": "6501.T", "name": "日立製作所", "sector": "機械"},
        {
            "code": "8306.T",
            "name": "三菱UFJフィナンシャルG",
            "sector": "銀行・金融業",
        },
        {"code": "4502.T", "name": "武田薬品工業", "sector": "医薬品"},
        {"code": "2914.T", "name": "JT", "sector": "食料品"},
        {"code": "8058.T", "name": "三菱商事", "sector": "商社・卸売業"},
        {"code": "9432.T", "name": "NTT", "sector": "情報通信・IT"},
        {"code": "6367.T", "name": "ダイキン工業", "sector": "機械"},
        {"code": "4063.T", "name": "信越化学工業", "sector": "化学・素材"},
        {"code": "7735.T", "name": "SCREENホールディングス", "sector": "半導体・電子部品"},
        {"code": "6146.T", "name": "ディスコ", "sector": "半導体・電子部品"},
        {"code": "8001.T", "name": "伊藤忠商事", "sector": "商社・卸売業"},
        {"code": "8316.T", "name": "三井住友フィナンシャルG", "sector": "銀行・金融業"},
        {"code": "7267.T", "name": "ホンダ", "sector": "自動車・輸送機器"},
    ]

    def __init__(self, min_turnover_jpy: float = 200_000_000):
        """初期化関数

        :param min_turnover_jpy: 【要件4】流動性基準の最低売買代金（デフォルト:
            2億円）
        """
        self.min_turnover_jpy = min_turnover_jpy

    def fetch_market_context(self) -> Dict[str, Any]:
        """【要件1】地合い（全体相場トレンド）とのリアルタイム完全連動

        日経平均株価（^N225）の乖離率および前日比を監視し、相場がパニック安（連れ安）状態か判定。
        """
        try:
            n225 = yf.Ticker("^N225").history(period="3mo")
            if len(n225) < 25:
                return {
                    "nikkei_price": 0.0,
                    "nikkei_kairi": 0.0,
                    "nikkei_change": 0.0,
                    "market_panic": False,
                }

            close = float(n225["Close"].iloc[-1])
            prev_close = float(n225["Close"].iloc[-2])
            ma25 = float(n225["Close"].tail(25).mean())

            nikkei_kairi = ((close - ma25) / ma25) * 100.0
            nikkei_change = ((close - prev_close) / prev_close) * 100.0

            # 地合いパニック判定: 日経平均25日乖離率 <= -3.5% または 日次落幅 <= -1.8%
            market_panic = (nikkei_kairi <= -3.5) or (nikkei_change <= -1.8)

            return {
                "nikkei_price": round(close, 2),
                "nikkei_kairi": round(nikkei_kairi, 2),
                "nikkei_change": round(nikkei_change, 2),
                "market_panic": market_panic,
            }
        except Exception:
            return {
                "nikkei_price": 0.0,
                "nikkei_kairi": 0.0,
                "nikkei_change": 0.0,
                "market_panic": False,
            }

    def fetch_and_filter_candidates(
        self, targets: List[Dict[str, str]]
    ) -> pd.DataFrame:
        """Step 1: データ収集 & 1次抽出 (`fetch_and_filter_candidates`)

        - 過去75日分以上の株価・出来高データを一括取得。
        - 流動性フィルター: 直近売買代金（終値 × 出来高）が2億円未満の銘柄を即座にカット。
        - MA25, MA50, MA75乖離率を算出。
        - MA25乖離率の下落率（マイナス乖離）が大きい順に1次抽出。
        """
        tickers = [item["code"] for item in targets]
        meta_lookup = {item["code"]: item for item in targets}

        # 過去75営業日以上（約4ヶ月分）のデータを取得
        data = yf.download(
            tickers, period="4mo", interval="1d", group_by="ticker", threads=True
        )

        candidates = []
        for code in tickers:
            meta = meta_lookup[code]
            try:
                if len(tickers) == 1:
                    df_stock = data.dropna(subset=["Close"])
                else:
                    if code not in data.columns.levels[0]:
                        continue
                    df_stock = data[code].dropna(subset=["Close"])

                if len(df_stock) < 75:
                    continue

                current_price = float(df_stock["Close"].iloc[-1])
                recent_volume = float(df_stock["Volume"].iloc[-1])
                daily_turnover = current_price * recent_volume

                # 【要件4】流動性フィルター: 直近売買代金が基準未満の銘柄を即座除外
                if daily_turnover < self.min_turnover_jpy:
                    continue

                # 【要件5】多角的な移動平均線（25日 / 50日 / 75日）の算出
                ma25 = float(df_stock["Close"].tail(25).mean())
                ma50 = float(df_stock["Close"].tail(50).mean())
                ma75 = float(df_stock["Close"].tail(75).mean())

                # 各乖離率 (%) の計算式: (現在値 - 移動平均) / 移動平均 * 100
                kairi25 = ((current_price - ma25) / ma25) * 100.0
                kairi50 = ((current_price - ma50) / ma50) * 100.0
                kairi75 = ((current_price - ma75) / ma75) * 100.0

                # 移動平均線の傾き (直近5日間の変化率%)
                ma25_5d_ago = float(df_stock["Close"].tail(30).head(25).mean())
                ma25_slope = ((ma25 - ma25_5d_ago) / ma25_5d_ago) * 100.0

                # 【要件4】出来高急増判定用の20日平均出来高
                ma20_vol = float(df_stock["Volume"].tail(20).mean())
                vol_ratio = (
                    (recent_volume / ma20_vol) if ma20_vol > 0 else 1.0
                )

                candidates.append({
                    "code": code.replace(".T", ""),
                    "full_code": code,
                    "name": meta["name"],
                    "sector": meta["sector"],
                    "price": round(current_price, 1),
                    "daily_turnover_oku": round(
                        daily_turnover / 100_000_000, 2
                    ),
                    "ma25": round(ma25, 1),
                    "ma50": round(ma50, 1),
                    "ma75": round(ma75, 1),
                    "ma25_slope": round(ma25_slope, 2),
                    "kairi25": round(kairi25, 2),
                    "kairi50": round(kairi50, 2),
                    "kairi75": round(kairi75, 2),
                    "vol_ratio": round(vol_ratio, 2),
                })
            except Exception:
                continue

        df_candidates = pd.DataFrame(candidates)
        if df_candidates.empty:
            return pd.DataFrame()

        # MA25乖離率が低い（マイナスが大きい）順にソートして1次抽出
        return df_candidates.sort_values(by="kairi25").reset_index(drop=True)

    def mechanical_filtering(
        self, df_candidates: pd.DataFrame
    ) -> pd.DataFrame:
        """Step 2: 機械的フィルター & セクター比較 (`mechanical_filtering`)

        - セクターごとの動態閾値を判定。
        - 出来高スパイク（`直近出来高 / 20日平均出来高 >= 2.0`）を測定。
        - 【要件3】個別悪材料（ナイフ）異常検知: セクター平均と比較して単独で突出して急落している銘柄を検知。
        """
        if df_candidates.empty:
            return df_candidates

        # セクターごとの平均MA25乖離率を動的に計算（連れ安 vs 単独急落の比較基準）
        sector_kairi_avg = (
            df_candidates.groupby("sector")["kairi25"].mean().to_dict()
        )

        processed = []
        for _, row in df_candidates.iterrows():
            sec_info = self.SECTOR_CONFIG.get(
                row["sector"],
                {
                    "threshold": self.DEFAULT_THRESHOLD,
                    "category": "一般セクター",
                },
            )
            threshold = sec_info["threshold"]
            sec_avg = sector_kairi_avg.get(row["sector"], row["kairi25"])

            # 1. セクター閾値クリアチェック
            passed_threshold = row["kairi25"] <= threshold

            # 2. 出来高スパイク判定 (2.0倍以上)
            has_volume_spike = row["vol_ratio"] >= 2.0

            # 3. 【要件3】単独急落・個別悪材料判定
            # (セクター平均乖離率よりさらに8.0%以上深く落ち込んでいる場合「単独急落ナイフ」と判定)
            isolated_drop_risk = (row["kairi25"] - sec_avg) < -8.0

            # セクター一斉の連れ安判定 (セクター平均乖離率が -4.0% 以下)
            sector_co_falling = sec_avg <= -4.0

            row_dict = row.to_dict()
            row_dict.update({
                "sector_threshold": threshold,
                "sector_category": sec_info["category"],
                "sector_avg_kairi": round(sec_avg, 2),
                "passed_threshold": passed_threshold,
                "has_volume_spike": has_volume_spike,
                "isolated_drop_risk": isolated_drop_risk,
                "sector_co_falling": sector_co_falling,
            })
            processed.append(row_dict)

        return pd.DataFrame(processed)

    def calculate_bnf_score(
        self, df_filtered: pd.DataFrame, market_ctx: Dict[str, Any]
    ) -> pd.DataFrame:
        """Step 3: 地合い同期 & BNF適合度スコアリング (`calculate_bnf_score`)

        各評価軸を定量スコア化し、合計0〜100点の「BNF適合度スコア」を算出。
        """
        if df_filtered.empty:
            return df_filtered

        results = []
        for _, row in df_filtered.iterrows():
            score = 0.0
            score_details = []

            # -----------------------------------------------------------------
            # 1. 【要件1】地合いスコア (最大20点)
            # -----------------------------------------------------------------
            if market_ctx.get("market_panic", False):
                score += 20.0
                score_details.append("全体相場パニック安時の連れ安 (+20pt)")
            elif market_ctx.get("nikkei_kairi", 0.0) < -1.5:
                score += 10.0
                score_details.append("全体地合い軟調 (+10pt)")
            else:
                score_details.append("全体相場平常・上昇傾向 (0pt)")

            # -----------------------------------------------------------------
            # 2. 【要件2/5】多角乖離スコア (最大35点)
            # -----------------------------------------------------------------
            k25 = row["kairi25"]
            thresh = row["sector_threshold"]

            # MA25乖離率スコア (最大20点)
            excess = thresh - k25  # 閾値超過分
            if excess > 0:
                m25_score = min(20.0, 10.0 + excess * 1.5)
            else:
                m25_score = max(0.0, 10.0 + excess * 1.0)
            score += m25_score

            # MA75乖離率スコア (最大15点)
            k75 = row["kairi75"]
            if k75 <= -20.0:
                score += 15.0
            elif k75 <= -12.0:
                score += 10.0
            elif k75 <= -7.0:
                score += 5.0

            # -----------------------------------------------------------------
            # 3. 【要件4】出来高急増（セリクラ）スコア (最大20点)
            # -----------------------------------------------------------------
            v_ratio = row["vol_ratio"]
            if v_ratio >= 3.0:
                score += 20.0
                score_details.append(
                    f"出来高クライマックス発生: {v_ratio:.1f}倍 (+20pt)"
                )
            elif v_ratio >= 2.0:
                score += 15.0
                score_details.append(f"出来高急増: {v_ratio:.1f}倍 (+15pt)")
            elif v_ratio >= 1.5:
                score += 8.0
                score_details.append(f"出来高増加: {v_ratio:.1f}倍 (+8pt)")

            # -----------------------------------------------------------------
            # 4. 【要件3】連れ安評価 / 単独急落ペナルティ (最大25点 / 減点-30点)
            # -----------------------------------------------------------------
            if row["isolated_drop_risk"]:
                # 個別ナイフリスクは致命的ペナルティ
                score -= 30.0
                score_details.append(
                    "⚠️ 警告: 単独突出急落・個別悪材料リスク (-30pt)"
                )
            elif row["sector_co_falling"]:
                score += 25.0
                score_details.append("セクター一斉パニック連れ安 (+25pt)")
            else:
                score += 10.0

            # スコア範囲を 0.0 〜 100.0 に正規化
            final_score = float(np.clip(score, 0.0, 100.0))

            row_dict = row.to_dict()
            row_dict["bnf_score"] = round(final_score, 1)
            row_dict["score_details"] = score_details
            results.append(row_dict)

        df_scored = pd.DataFrame(results)
        return df_scored.sort_values(by="bnf_score", ascending=False).reset_index(
            drop=True
        )

    def generate_final_report(
        self, df_scored: pd.DataFrame, market_ctx: Dict[str, Any], top_n: int = 5
    ) -> str:
        """Step 4: 最終出力UI & レポート生成 (`generate_final_report`)

        可読性の高いフォーマットで可視化レポートを作成。
        """
        output = []
        output.append(
            "================================================================================"
        )
        output.append(
            " 🏛️ BNF流・自動逆張りスイングトレード スクリーニング ＆ スコアリング レポート"
        )
        output.append(
            "================================================================================"
        )
        output.append(
            f" 📅 スキャン実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        output.append(" 📈 全体地合い診断 (Market Context):")
        output.append(
            f"   ・日経平均株価: {market_ctx.get('nikkei_price', 0.0):,} 円"
            f" (前日比: {market_ctx.get('nikkei_change', 0.0):+.2f}%)"
        )
        output.append(
            f"   ・日経平均25日乖離率: {market_ctx.get('nikkei_kairi', 0.0):+.2f}%"
        )

        panic_status = (
            "🔥 パニック安発生（逆張り期待値【高】）"
            if market_ctx.get("market_panic")
            else "🟢 相場平常/安定（逆張り評価判定【厳格化】）"
        )
        output.append(f"   ・地合い判定: {panic_status}")
        output.append(
            "================================================================================"
        )
        output.append("")

        if df_scored.empty:
            output.append(
                "⚠️ 条件（流動性・売買代金・乖離率）を満たす候補銘柄は見つかりませんでした。"
            )
            return "\n".join(output)

        top_candidates = df_scored.head(top_n)

        output.append(
            f"【 🎯 BNF適合度スコア 厳選上位 {len(top_candidates)} 銘柄 】\n"
        )

        for idx, (_, row) in enumerate(top_candidates.iterrows(), start=1):
            output.append(
                "────────────────────────────────────────────────────────────────────────────────"
            )
            output.append(
                f"第 {idx} 位 : [{row['code']}] {row['name']}  │ BNF適合度スコア: {row['bnf_score']} / 100 点"
            )
            output.append(
                "────────────────────────────────────────────────────────────────────────────────"
            )
            output.append(
                f" ・セクター区分   : {row['sector']} ({row['sector_category']})"
            )
            output.append(
                f" ・現在株価       : {row['price']:,} 円  (日平均売買代金:"
                f" {row['daily_turnover_oku']} 億円)"
            )
            output.append(
                f" ・25日線乖離率   : {row['kairi25']:+.2f}%  [セクター動態閾値:"
                f" {row['sector_threshold']:+.1f}%]"
            )
            output.append(
                f" ・75日線乖離率   : {row['kairi75']:+.2f}%  (50日線乖離率:"
                f" {row['kairi50']:+.2f}%)"
            )
            output.append(
                f" ・出来高倍率     : {row['vol_ratio']:.2f} 倍 (20日平均比)"
            )
            output.append(
                f" ・セクター平均乖離: {row['sector_avg_kairi']:+.2f}%"
            )
            output.append("")

            # 逆張り理由の分析自動生成
            reasons = []
            if row["sector_co_falling"]:
                reasons.append(
                    f"セクター全体の連れ安（平均乖離 {row['sector_avg_kairi']:+.1f}%）によるパニック売り。"
                )
            if row["has_volume_spike"]:
                reasons.append(
                    f"出来高が通常の {row['vol_ratio']:.1f} 倍に急増しており、投げ売り（セリクラ）完了の示唆。"
                )
            if row["kairi25"] <= row["sector_threshold"]:
                reasons.append(
                    f"25日線乖離率（{row['kairi25']:+.1f}%）がセクター固有の動態閾値（{row['sector_threshold']:+.1f}%）を超過。"
                )

            reason_text = (
                " ".join(reasons)
                if reasons
                else "一定の売られすぎ水準に達している。"
            )
            output.append(" 💡 選定・逆張り理由 (ロジック解説):")
            output.append(f"    {reason_text}")

            # リスク注意点の自動生成
            risks = []
            if row["isolated_drop_risk"]:
                risks.append(
                    "⚠️ 【警告】セクター平均に対し単独で突出急落中。個別悪材料（決算下方修正・不祥事等）の落ちてくるナイフに注意。"
                )
            if row["kairi75"] < -25.0:
                risks.append(
                    "⚠️ 中長期トレンド（75日線）が強い下落トレンド。深追いを避け、1〜3営業日での短期手仕舞いを徹底。"
                )
            if not market_ctx.get("market_panic"):
                risks.append(
                    "⚠️ 全体相場がパニック安状態ではないため、連れ安反発の推進力がやや弱い可能性あり。"
                )

            risk_text = (
                "\n    ".join(risks)
                if risks
                else "特筆すべき不自然な個別悪材料なし（正常な地合い連れ安）。"
            )
            output.append(" ⚠️ リスク注意点・トレード指針:")
            output.append(f"    {risk_text}")
            output.append("")

        output.append(
            "================================================================================"
        )
        output.append(" 📌 BNFトレード運用指針:")
        output.append(
            "  1. エントリー: 寄付き直後のパニック売り気配（寄り底狙い）、または前場引けでの押し目買い。"
        )
        output.append(
            "  2. 利益確定   : 25日線乖離率が -3.0% 〜 0.0% に復帰した段階で機械的に利益確定（保有期間1〜3日）。"
        )
        output.append(
            "  3. 損切り     : 買値から -2.5% 〜 -3.0% を下回った場合は迷わず即時撤退。"
        )
        output.append(
            "================================================================================"
        )

        return "\n".join(output)

    def run_pipeline(
        self, targets: List[Dict[str, str]] = None
    ) -> Tuple[str, pd.DataFrame]:
        """全4段階パイプラインを順次実行してレポートを出力"""
        if targets is None:
            targets = self.SAMPLE_TARGETS

        print("[Step 1] 全体地合いデータの取得中...")
        market_ctx = self.fetch_market_context()

        print(
            f"[Step 1] 対象 {len(targets)} 銘柄の株価取得 & 流動性（売買代金2億円以上）フィルター適用中..."
        )
        df_candidates = self.fetch_and_filter_candidates(targets)

        print("[Step 2] セクター動態閾値判定 & 単独急落ナイフ検知フィルター適用中...")
        df_filtered = self.mechanical_filtering(df_candidates)

        print("[Step 3] 地合い連動 & BNF適合度（0〜100点）自動スコアリング算出中...")
        df_scored = self.calculate_bnf_score(df_filtered, market_ctx)

        print("[Step 4] 最終レポートUI生成中...\n")
        report_text = self.generate_final_report(df_scored, market_ctx)

        return report_text, df_scored


# ═════════════════════════════════════════════════════════════════════════════
# メイン実行処理
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 最低売買代金 2億円（200,000,000円）でスキャナーを初期化
    scanner = BNFTradingScanner(min_turnover_jpy=200_000_000)

    # 4段階パイプラインの実行
    report, results_df = scanner.run_pipeline()

    # レポート表示
    print(report)
