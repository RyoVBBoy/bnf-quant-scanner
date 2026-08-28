import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Any, Tuple

class BNFTradingScanner:
    """
    BNF流・東証主力高流動性銘柄 厳選スキャンエンジン
    """
    
    SECTOR_CONFIG: Dict[str, Dict[str, Any]] = {
        "自動車・輸送機器": {"threshold": -10.0, "category": "大型・バリュー"},
        "機械・プラント": {"threshold": -10.0, "category": "大型・バリュー"},
        "銀行・金融・保険": {"threshold": -10.0, "category": "大型・金融バリュー"},
        "商社・卸売": {"threshold": -10.0, "category": "大型・バリュー"},
        "鉄鋼・素材・化学": {"threshold": -10.0, "category": "シクリカル・バリュー"},
        "半導体・電子部品": {"threshold": -12.0, "category": "ハイテク・半導体"},
        "情報通信・IT・ネット": {"threshold": -15.0, "category": "ハイテク・IT"},
        "精密機器・医療機器": {"threshold": -12.0, "category": "ハイテク・精密"},
        "ゲーム・エンタメ": {"threshold": -12.0, "category": "ハイテク・コンテンツ"},
        "新興グロース・サービス": {"threshold": -20.0, "category": "新興・グロース"},
        "医薬品": {"threshold": -8.0, "category": "ディフェンシブ"},
        "食料品・消費財": {"threshold": -7.0, "category": "ディフェンシブ"},
        "陸運・海運・物流": {"threshold": -8.0, "category": "ディフェンシブ"},
        "電力・ガス・インフラ": {"threshold": -7.0, "category": "ディフェンシブ"},
        "小売・外食": {"threshold": -10.0, "category": "消費・ディフェンシブ"}
    }
    
    CORE_TARGETS: List[Dict[str, str]] = [
        {"code": "6920.T", "name": "レーザーテック", "sector": "半導体・電子部品"},
        {"code": "8035.T", "name": "東京エレクトロン", "sector": "半導体・電子部品"},
        {"code": "6857.T", "name": "アドバンテスト", "sector": "半導体・電子部品"},
        {"code": "6146.T", "name": "ディスコ", "sector": "半導体・電子部品"},
        {"code": "6758.T", "name": "ソニーG", "sector": "ゲーム・エンタメ"},
        {"code": "9984.T", "name": "ソフトバンクG", "sector": "情報通信・IT・ネット"},
        {"code": "7203.T", "name": "トヨタ自動車", "sector": "自動車・輸送機器"},
        {"code": "7267.T", "name": "ホンダ", "sector": "自動車・輸送機器"},
        {"code": "8306.T", "name": "三菱UFJ", "sector": "銀行・金融・保険"},
        {"code": "8316.T", "name": "三井住友", "sector": "銀行・金融・保険"},
        {"code": "8411.T", "name": "みずほ", "sector": "銀行・金融・保険"},
        {"code": "8058.T", "name": "三菱商事", "sector": "商社・卸売"},
        {"code": "8001.T", "name": "伊藤忠", "sector": "商社・卸売"},
        {"code": "8031.T", "name": "三井物産", "sector": "商社・卸売"},
        {"code": "9101.T", "name": "日本郵船", "sector": "陸運・海運・物流"},
        {"code": "9104.T", "name": "商船三井", "sector": "陸運・海運・物流"},
        {"code": "9107.T", "name": "川崎汽船", "sector": "陸運・海運・物流"},
        {"code": "7974.T", "name": "任天堂", "sector": "ゲーム・エンタメ"},
        {"code": "9983.T", "name": "ファーストリテイリング", "sector": "小売・外食"},
        {"code": "6098.T", "name": "リクルートHD", "sector": "新興グロース・サービス"},
        {"code": "4385.T", "name": "メルカリ", "sector": "新興グロース・サービス"},
        {"code": "6501.T", "name": "日立製作所", "sector": "機械・プラント"},
        {"code": "6367.T", "name": "ダイキン工業", "sector": "機械・プラント"},
        {"code": "4063.T", "name": "信越化学", "sector": "鉄鋼・素材・化学"},
        {"code": "5401.T", "name": "日本製鉄", "sector": "鉄鋼・素材・化学"},
        {"code": "4502.T", "name": "武田薬品", "sector": "医薬品"},
        {"code": "4519.T", "name": "中外製薬", "sector": "医薬品"},
        {"code": "2914.T", "name": "JT", "sector": "食料品・消費財"},
        {"code": "9432.T", "name": "NTT", "sector": "情報通信・IT・ネット"},
        {"code": "9433.T", "name": "KDDI", "sector": "情報通信・IT・ネット"},
        {"code": "9501.T", "name": "東京電力HD", "sector": "電力・ガス・インフラ"}
    ]

    def __init__(self, min_turnover_jpy: float = 500_000_000):
        self.min_turnover_jpy = min_turnover_jpy

    def fetch_market_context(self) -> Dict[str, Any]:
        try:
            n225 = yf.Ticker("^N225").history(period="2mo")
            if len(n225) < 25:
                return {"nikkei_price": 0, "nikkei_kairi": 0.0, "nikkei_change": 0.0, "market_panic": False}
            
            close = float(n225['Close'].iloc[-1])
            prev_close = float(n225['Close'].iloc[-2])
            ma25 = float(n225['Close'].tail(25).mean())
            
            kairi = ((close - ma25) / ma25) * 100.0
            chg = ((close - prev_close) / prev_close) * 100.0
            
            return {
                "nikkei_price": round(close, 2),
                "nikkei_kairi": round(kairi, 2),
                "nikkei_change": round(chg, 2),
                "market_panic": bool(kairi <= -3.5 or chg <= -1.8)
            }
        except Exception:
            return {"nikkei_price": 0, "nikkei_kairi": 0.0, "nikkei_change": 0.0, "market_panic": False}

    def run_pipeline(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        market_ctx = self.fetch_market_context()
        tickers = [item["code"] for item in self.CORE_TARGETS]
        meta_lookup = {item["code"]: item for item in self.CORE_TARGETS}
        
        data = yf.download(tickers, period="3mo", interval="1d", group_by='ticker', threads=True)
        results = []

        for code in tickers:
            try:
                df = data[code].dropna(subset=['Close']) if len(tickers) > 1 else data.dropna(subset=['Close'])
                if len(df) < 25: continue
                
                price = float(df['Close'].iloc[-1])
                vol = float(df['Volume'].iloc[-1])
                turnover = price * vol
                
                if turnover < self.min_turnover_jpy: continue
                
                ma25 = float(df['Close'].tail(25).mean())
                ma75 = float(df['Close'].tail(75).mean()) if len(df) >= 75 else ma25
                
                kairi25 = ((price - ma25) / ma25) * 100.0
                kairi75 = ((price - ma75) / ma75) * 100.0
                
                ma20_vol = float(df['Volume'].tail(20).mean())
                vol_ratio = (vol / ma20_vol) if ma20_vol > 0 else 1.0
                
                meta = meta_lookup[code]
                sec_cfg = self.SECTOR_CONFIG.get(meta["sector"], {"threshold": -12.0})
                
                score = 50.0 + (sec_cfg["threshold"] - kairi25) * 2.0
                if vol_ratio >= 2.0: score += 15.0
                if market_ctx["market_panic"]: score += 15.0
                score = float(np.clip(score, 0.0, 100.0))

                results.append({
                    "code": code.replace(".T", ""),
                    "name": meta["name"],
                    "sector": meta["sector"],
                    "price": round(price, 1),
                    "kairi25": round(kairi25, 2),
                    "kairi75": round(kairi75, 2),
                    "vol_ratio": round(vol_ratio, 2),
                    "sector_threshold": sec_cfg["threshold"],
                    "bnf_score": round(score, 1)
                })
            except Exception:
                continue

        results.sort(key=lambda x: x["bnf_score"], reverse=True)
        return market_ctx, results
