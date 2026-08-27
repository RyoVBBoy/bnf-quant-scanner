from flask import Flask, render_template_string
from bnf_quant_scanner import BNFTradingScanner

app = Flask(__name__)

# 完全カスタマイズ可能なHTML/CSSデザイン
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>BNF Quant Scanner</title>
    <style>
        :root { --bg: #0d1117; --card-bg: #161b22; --border: #30363d; --text: #c9d1d9; --accent: #58a6ff; --green: #238636; --orange: #f0883e; --red: #da3633; }
        body { background-color: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 24px; }
        .container { max-width: 1000px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: 16px; margin-bottom: 24px; }
        h1 { color: var(--accent); margin: 0; font-size: 24px; }
        .btn { background-color: var(--green); color: #fff; border: none; padding: 10px 20px; border-radius: 6px; font-weight: bold; cursor: pointer; text-decoration: none; display: inline-block; }
        .btn:hover { opacity: 0.9; }
        .market-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 24px; display: flex; gap: 24px; }
        .metric { display: flex; flex-direction: column; }
        .metric-label { font-size: 12px; color: #8b949e; }
        .metric-value { font-size: 20px; font-weight: bold; margin-top: 4px; }
        .stock-card { background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 16px; }
        .stock-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }
        .stock-title { font-size: 18px; font-weight: bold; color: #fff; }
        .score-badge { background: rgba(240, 136, 62, 0.15); color: var(--orange); border: 1px solid var(--orange); padding: 4px 12px; border-radius: 20px; font-weight: bold; }
        .grid { display: grid; grid-template-columns: repeat(4, 1dfr); gap: 12px; margin-bottom: 12px; background: #0d1117; padding: 12px; border-radius: 6px; }
        .box-title { font-size: 11px; color: #8b949e; }
        .box-val { font-size: 14px; font-weight: bold; margin-top: 2px; }
        .reason-box { background: rgba(88, 166, 255, 0.08); border-left: 3px solid var(--accent); padding: 10px; margin-top: 10px; font-size: 13px; line-height: 1.5; }
        .risk-box { background: rgba(218, 54, 51, 0.08); border-left: 3px solid var(--red); padding: 10px; margin-top: 8px; font-size: 13px; color: #ff7b72; line-height: 1.5; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏛️ BNF流 自動逆張りスキャナー</h1>
            <a href="/run" class="btn">🚀 今すぐスキャン実行</a>
        </div>

        {% if market %}
        <div class="market-card">
            <div class="metric">
                <span class="metric-label">日経平均株価</span>
                <span class="metric-value">{{ market.nikkei_price }} 円</span>
            </div>
            <div class="metric">
                <span class="metric-label">前日比</span>
                <span class="metric-value" style="color: {{ 'var(--red)' if market.nikkei_change < 0 else 'var(--green)' }}">{{ market.nikkei_change }}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">25日線乖離率</span>
                <span class="metric-value">{{ market.nikkei_kairi }}%</span>
            </div>
            <div class="metric">
                <span class="metric-label">全体地合い状態</span>
                <span class="metric-value" style="color: var(--accent)">{{ '🔥 パニック安' if market.market_panic else '🟢 平常・安定' }}</span>
            </div>
        </div>

        <h2 style="font-size: 18px; margin-bottom: 16px;">🎯 BNF適合度 厳選上位銘柄</h2>

        {% for item in stocks %}
        <div class="stock-card">
            <div class="stock-header">
                <div class="stock-title">第 {{ loop.index }} 位 : [{{ item.code }}] {{ item.name }} <span style="font-size: 12px; color: #8b949e; margin-left: 8px;">{{ item.sector }}</span></div>
                <div class="score-badge">BNFスコア: {{ item.bnf_score }} / 100</div>
            </div>
            <div class="grid">
                <div><div class="box-title">現在株価</div><div class="box-val">{{ item.price }} 円</div></div>
                <div><div class="box-title">25日線乖離率</div><div class="box-val" style="color: var(--orange)">{{ item.kairi25 }}%</div></div>
                <div><div class="box-title">セクター動態閾値</div><div class="box-val">{{ item.sector_threshold }}%</div></div>
                <div><div class="box-title">出来高倍率</div><div class="box-val">{{ item.vol_ratio }} 倍</div></div>
            </div>
            <div class="reason-box">
                <strong>💡 選定・逆張り理由:</strong><br>
                25日線乖離率が閾値突破。{{ 'セクター一斉の連れ安が発生中。' if item.sector_co_falling else '' }} {{ '出来高が通常の ' ~ item.vol_ratio ~ ' 倍に急増（セリクラ検知）。' if item.has_volume_spike else '' }}
            </div>
            {% if item.isolated_drop_risk or item.kairi75 < -25.0 %}
            <div class="risk-box">
                <strong>⚠️ リスク注意点:</strong><br>
                {{ 'セクター平均に対し単独で急落中（個別悪材料リスク注意）。' if item.isolated_drop_risk else '' }}
                {{ '75日線乖離率が大きいため短打（1〜3日）徹底。' if item.kairi75 < -25.0 else '' }}
            </div>
            {% endif %}
        </div>
        {% endfor %}
        {% else %}
        <p style="text-align: center; color: #8b949e; margin-top: 60px;">「今すぐスキャン実行」ボタンを押すと、リアルタイムで自動解析が始まります。</p>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_PAGE, market=None, stocks=[])

@app.route('/run')
def run():
    scanner = BNFTradingScanner(min_turnover_jpy=200_000_000)
    market_ctx, results = scanner.run_pipeline()
    return render_template_string(HTML_PAGE, market=market_ctx, stocks=results[:5])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
