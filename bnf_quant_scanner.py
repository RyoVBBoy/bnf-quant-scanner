import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io
from datetime import datetime
from typing import Dict, List, Any, Tuple

class BNFTradingScanner:
    """
    BNF流・自動逆張りスキャニング＆スコアリングエンジン
    """
    
    SECTOR_CONFIG: Dict[str, Dict[str, Any]] = {
        "自動車・輸送機器": {"threshold": -10.0, "category": "大型・バリュー"},
        "機械": {"threshold": -10.0, "category": "大型・バリュー"},
        "銀行・金融業": {"threshold": -10.0, "category": "大型・金融バリュー"},
        "商社・卸売業": {"threshold": -10.0, "category": "大型・バリュー"},
        "鉄鋼・非鉄金属": {"threshold": -10.0, "category": "シクリカル・バリュー"},
        "化学・素材": {"threshold": -10.0, "category": "シクリカル・バリュー"},
        "半導体・電子部品": {"threshold": -12.0, "category": "ハイテク・半導体"},
        "情報通信・IT": {"threshold": -15.0, "category": "ハイテク・IT"},
        "精密機器": {"threshold": -12.0, "category": "ハイテク・精密"},
        "ゲーム・エンタメ": {"threshold": -12.0, "category": "ハイテク・コンテンツ"},
        "サービス・グロース": {"threshold": -20.0, "category": "新興・グロース"},
        "ネットEC・プラットフォーム": {"threshold": -22.0, "category": "新興・グロース"},
        "医薬品": {"threshold": -8.0, "category": "ディフェンシブ"},
        "食料品": {"threshold": -7.0, "category": "ディフェンシブ"},
        "陸運・物流": {"threshold": -8.0, "category": "ディフェンシブ"},
        "電気・ガス業": {"threshold": -7.0, "category": "ディフェンシブ"},
        "小売・消費": {"threshold": -10.0, "category": "消費・ディフェンシブ"}
    }
    
    DEFAULT_THRESHOLD: float = -12.0

    SAMPLE_TARGETS: List[Dict[str, str]] = [
        {"code": "6920.T", "name": "レーザーテック", "sector": "半導体・電子部品"},
        {"code": "8035.T", "name": "東京エレクトロン", "sector": "半導体・電子部品"},
        {"code": "6857.T", "name": "アドバンテスト", "sector": "半導体・電子部品"},
        {"code": "9984.T", "name": "ソフトバンクG", "sector": "情報通信・IT"},
        {"code": "6758.T", "name": "ソニーG", "sector": "ゲーム・エンタメ"},
        {"code": "4385.T", "name": "メルカリ", "sector": "ネットEC・プラットフォーム"},
        {"code": "9101.T", "name": "日本郵船", "sector": "陸運・物流"},
        {"code": "7974.T", "name": "任天堂", "sector": "ゲーム・エンタメ"},
        {"code": "6981.T", "name": "村田製作所", "sector": "半導体・電子部品"},
        {"code": "6098.T", "name": "リクルートHD", "sector": "サービス・グロース"},
        {"code": "7203.T", "name": "トヨタ自動車", "sector": "自動車・輸送機器"},
        {"code": "9983.T", "name": "ファーストリテイリング", "sector": "小売・消費"},
        {"code": "6501.T", "name": "日立製作所", "sector": "機械"},
        {"code": "8306.T", "name": "三菱UFJフィナンシャルG", "sector": "銀行・金融業"},
        {"code": "4502.T", "name": "武田薬品工業", "sector": "医薬品"},
        {"code": "2914.T", "name": "JT", "sector": "食料品"},
        {"code": "8058.T", "name": "三菱商事", "sector": "商社・卸売業"},
        {"code": "9432.T", "name": "NTT", "sector": "情報通信・IT"},
        {"code": "6367.T", "name": "ダイキン工業", "sector": "機械"},
        {"code": "4063.T", "name": "信越化学工業", "sector": "化学・素材"}
    ]

    def __init__(self, min_turnover_jpy: float = 200_000_000):
        self.min_turnover_jpy = min_turnover_jpy

    def fetch_market_context(self) -> Dict[str, Any]:
        try:
            n225 = yf.Ticker("^N225").history(period="3mo")
            if len(n225) < 25:
                return {"nikkei_price": 0.0, "nikkei_kairi": 0.0, "nikkei_change": 0.0, "market_panic": False}
            
            close = float(n225['Close'].iloc[-1])
            prev_close = float(n225['Close'].iloc[-2])
            ma25 = float(n225['Close'].tail(25).mean())
            
            nikkei_kairi = ((close - ma25) / ma25) * 100.0
            nikkei_change = ((close - prev_close) / prev_close) * 100.0
            market_panic = (nikkei_kairi <= -3.5) or (nikkei_change <= -1.8)
            
            return {
                "nikkei_price": round(close, 2),
                "nikkei_kairi": round(nikkei_kairi, 2),
                "nikkei_change": round(nikkei_change, 2),
                "market_panic": market_panic
            }
        except Exception:
            return {"nikkei_price": 0.0, "nikkei_kairi": 0.0, "nikkei_change": 0.0, "market_panic": False}

    def fetch_and_filter_candidates(self, targets: List[Dict[str, str]]) -> pd.DataFrame:
        tickers = [item["code"] for item in targets]
        meta_lookup = {item["code"]: item for item in targets}
        
        data = yf.download(tickers, period="4mo", interval="1d", group_by='ticker', threads=True)
        candidates = []
        
        for code in tickers:
            meta = meta_lookup[code]
            try:
                df_stock = data[code].dropna(subset=['Close']) if len(tickers) > 1 else data.dropna(subset=['Close'])
                if len(df_stock) < 75:
                    continue
                
                current_price = float(df_stock['Close'].iloc[-1])
                recent_volume = float(df_stock['Volume'].iloc[-1])
                daily_turnover = current_price * recent_volume
                
                if daily_turnover < self.min_turnover_jpy:
                    continue
                
                ma25 = float(df_stock['Close'].tail(25).mean())
                ma50 = float(df_stock['Close'].tail(50).mean())
                ma75 = float(df_stock['Close'].tail(75).mean())
                
                kairi25 = ((current_price - ma25) / ma25) * 100.0
                kairi50 = ((current_price - ma50) / ma50) * 100.0
                kairi75 = ((current_price - ma75) / ma75) * 100.0
                
                ma20_vol = float(df_stock['Volume'].tail(20).mean())
                vol_ratio = (recent_volume / ma20_vol) if ma20_vol > 0 else 1.0
                
                candidates.append({
                    "code": code.replace(".T", ""),
                    "full_code": code,
                    "name": meta["name"],
                    "sector": meta["sector"],
                    "price": round(current_price, 1),
                    "daily_turnover_oku": round(daily_turnover / 100_000_000, 2),
                    "ma25": round(ma25, 1),
                    "ma50": round(ma50, 1),
                    "ma75": round(ma75, 1),
                    "kairi25": round(kairi25, 2),
                    "kairi50": round(kairi50, 2),
                    "kairi75": round(kairi75, 2),
                    "vol_ratio": round(vol_ratio, 2)
                })
            except Exception:
                continue
                
        df_candidates = pd.DataFrame(candidates)
        return df_candidates.sort_values(by="kairi25").reset_index(drop=True) if not df_candidates.empty else pd.DataFrame()

    def mechanical_filtering(self, df_candidates: pd.DataFrame) -> pd.DataFrame:
        if df_candidates.empty:
            return df_candidates

        sector_kairi_avg = df_candidates.groupby("sector")["kairi25"].mean().to_dict()
        processed = []
        
        for _, row in df_candidates.iterrows():
            sec_info = self.SECTOR_CONFIG.get(row["sector"], {"threshold": self.DEFAULT_THRESHOLD, "category": "一般"})
            threshold = sec_info["threshold"]
            sec_avg = sector_kairi_avg.get(row["sector"], row["kairi25"])
            
            passed_threshold = row["kairi25"] <= threshold
            has_volume_spike = row["vol_ratio"] >= 2.0
            isolated_drop_risk = (row["kairi25"] - sec_avg) < -8.0
            sector_co_falling = sec_avg <= -4.0
            
            row_dict = row.to_dict()
            row_dict.update({
                "sector_threshold": threshold,
                "sector_category": sec_info["category"],
                "sector_avg_kairi": round(sec_avg, 2),
                "passed_threshold": passed_threshold,
                "has_volume_spike": has_volume_spike,
                "isolated_drop_risk": isolated_drop_risk,
                "sector_co_falling": sector_co_falling
            })
            processed.append(row_dict)

        return pd.DataFrame(processed)

    def calculate_bnf_score(self, df_filtered: pd.DataFrame, market_ctx: Dict[str, Any]) -> pd.DataFrame:
        if df_filtered.empty:
            return df_filtered

        results = []
        for _, row in df_filtered.iterrows():
            score = 0.0
            score_details = []

            if market_ctx.get("market_panic", False):
                score += 20.0
                score_details.append("全体地合いパニック安 (+20pt)")
            elif market_ctx.get("nikkei_kairi", 0.0) < -1.5:
                score += 10.0
                score_details.append("全体地合い軟調 (+10pt)")

            k25 = row["kairi25"]
            thresh = row["sector_threshold"]
            excess = thresh - k25
            score += min(20.0, 10.0 + excess * 1.5) if excess > 0 else max(0.0, 10.0 + excess * 1.0)
            
            k75 = row["kairi75"]
            if k75 <= -20.0: score += 15.0
            elif k75 <= -12.0: score += 10.0
            elif k75 <= -7.0: score += 5.0

            v_ratio = row["vol_ratio"]
            if v_ratio >= 3.0: score += 20.0
            elif v_ratio >= 2.0: score += 15.0
            elif v_ratio >= 1.5: score += 8.0

            if row["isolated_drop_risk"]:
                score -= 30.0
                score_details.append("単独急落リスク (-30pt)")
            elif row["sector_co_falling"]:
                score += 25.0
                score_details.append("セクター一斉連れ安 (+25pt)")

            row_dict = row.to_dict()
            row_dict["bnf_score"] = round(float(np.clip(score, 0.0, 100.0)), 1)
            row_dict["score_details"] = score_details
            results.append(row_dict)

        df_scored = pd.DataFrame(results)
        return df_scored.sort_values(by="bnf_score", ascending=False).reset_index(drop=True)

    def run_pipeline(self, targets: List[Dict[str, str]] = None) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        if targets is None:
            targets = self.SAMPLE_TARGETS
        market_ctx = self.fetch_market_context()
        df_candidates = self.fetch_and_filter_candidates(targets)
        df_filtered = self.mechanical_filtering(df_candidates)
        df_scored = self.calculate_bnf_score(df_filtered, market_ctx)
        
        results_list = df_scored.to_dict(orient="records") if not df_scored.empty else []
        return market_ctx, results_list
