from flask import Flask, send_from_directory, jsonify
from bnf_quant_scanner import BNFTradingScanner

app = Flask(__name__, static_folder='.')

@app.route('/')
def serve_index():
    # あなたの index.html をデザイン崩れゼロで100%そのまま表示
    return send_from_directory('.', 'index.html')

@app.route('/api/scan')
def api_scan():
    # 裏側で計算結果だけをJSONデータとして送信
    scanner = BNFTradingScanner()
    market_ctx, results = scanner.run_pipeline()
    return jsonify({"market": market_ctx, "results": results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
