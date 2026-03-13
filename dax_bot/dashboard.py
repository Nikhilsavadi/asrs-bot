"""
dashboard.py -- ASRS Performance Dashboard
==============================================================================

Flask web UI showing equity curve, trade log, slippage analysis, and stats.
Runs on port 8080 in a background thread from bot.py.
"""

import logging
import threading
from flask import Flask, jsonify, render_template_string

from dax_bot import config
from dax_bot.journal import get_stats, load_all_trades

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASRS Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
               background: #0d1117; color: #c9d1d9; padding: 20px; }
        h1 { color: #58a6ff; margin-bottom: 5px; }
        .subtitle { color: #8b949e; margin-bottom: 20px; font-size: 14px; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                      gap: 12px; margin-bottom: 24px; }
        .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                     padding: 16px; text-align: center; }
        .stat-value { font-size: 28px; font-weight: bold; color: #f0f6fc; }
        .stat-value.green { color: #3fb950; }
        .stat-value.red { color: #f85149; }
        .stat-label { font-size: 12px; color: #8b949e; margin-top: 4px; text-transform: uppercase; }
        .chart-container { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                          padding: 20px; margin-bottom: 24px; }
        .chart-container h2 { color: #58a6ff; font-size: 16px; margin-bottom: 12px; }
        .charts-row { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px; }
        @media (max-width: 900px) { .charts-row { grid-template-columns: 1fr; } }
        table { width: 100%; border-collapse: collapse; font-size: 13px; }
        th { background: #21262d; color: #8b949e; text-align: left; padding: 10px 12px;
             border-bottom: 1px solid #30363d; position: sticky; top: 0; cursor: pointer; }
        th:hover { color: #58a6ff; }
        td { padding: 8px 12px; border-bottom: 1px solid #21262d; }
        tr:hover td { background: #161b22; }
        .pnl-pos { color: #3fb950; }
        .pnl-neg { color: #f85149; }
        .slip-bad { color: #f85149; }
        .slip-ok { color: #8b949e; }
        .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; }
        .badge-long { background: #1a4731; color: #3fb950; }
        .badge-short { background: #4a1a1a; color: #f85149; }
        .badge-tp { background: #1a3a4a; color: #58a6ff; }
        .no-data { text-align: center; padding: 60px; color: #8b949e; font-size: 18px; }
        .monthly-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
                       gap: 8px; }
        .monthly-card { background: #21262d; border-radius: 6px; padding: 10px; text-align: center; }
        .monthly-card .month { font-size: 12px; color: #8b949e; }
        .monthly-card .val { font-size: 18px; font-weight: bold; margin-top: 4px; }
        .refresh-btn { float: right; background: #21262d; color: #8b949e; border: 1px solid #30363d;
                      padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }
        .refresh-btn:hover { border-color: #58a6ff; color: #58a6ff; }
    </style>
</head>
<body>
    <button class="refresh-btn" onclick="location.reload()">Refresh</button>
    <h1>ASRS DAX Trading Dashboard</h1>
    <p class="subtitle" id="mode"></p>

    <div id="content">
        <div class="no-data">Loading...</div>
    </div>

    <script>
    const MODE = "{{ mode }}";
    document.getElementById('mode').textContent =
        MODE + ' | ' + {{ num_contracts }} + 'x Micro DAX | TP1=+{{ tp1 }} TP2=+{{ tp2 }}';

    fetch('/api/stats')
        .then(r => r.json())
        .then(data => render(data))
        .catch(e => {
            document.getElementById('content').innerHTML =
                '<div class="no-data">Error loading data: ' + e + '</div>';
        });

    function render(s) {
        if (s.total_trades === 0) {
            document.getElementById('content').innerHTML =
                '<div class="no-data">No trades yet. Dashboard will populate after first trading day.</div>';
            return;
        }

        const pnlClass = s.total_pnl >= 0 ? 'green' : 'red';
        const pnlSign = s.total_pnl >= 0 ? '+' : '';

        let html = `
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value ${pnlClass}">${pnlSign}${s.total_pnl}</div>
                <div class="stat-label">Total P&L (pts)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${s.total_trades}</div>
                <div class="stat-label">Total Trades</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${s.win_rate}%</div>
                <div class="stat-label">Win Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${s.expectancy}</div>
                <div class="stat-label">Expectancy (pts)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${s.profit_factor}</div>
                <div class="stat-label">Profit Factor</div>
            </div>
            <div class="stat-card">
                <div class="stat-value red">${s.max_drawdown}</div>
                <div class="stat-label">Max Drawdown (pts)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value ${s.avg_slippage > 1 ? 'red' : ''}">${s.avg_slippage}</div>
                <div class="stat-label">Avg Slippage (pts)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${s.trading_days}</div>
                <div class="stat-label">Trading Days</div>
            </div>
        </div>

        <div class="charts-row">
            <div class="chart-container">
                <h2>Equity Curve</h2>
                <canvas id="equityChart"></canvas>
            </div>
            <div class="chart-container">
                <h2>Drawdown</h2>
                <canvas id="ddChart"></canvas>
            </div>
        </div>

        <div class="charts-row">
            <div class="chart-container">
                <h2>Daily P&L</h2>
                <canvas id="dailyChart"></canvas>
            </div>
            <div class="chart-container">
                <h2>Slippage per Trade</h2>
                <canvas id="slipChart"></canvas>
            </div>
        </div>`;

        // Monthly breakdown
        const months = Object.entries(s.monthly_pnl);
        if (months.length > 0) {
            html += '<div class="chart-container"><h2>Monthly P&L</h2><div class="monthly-grid">';
            for (const [m, v] of months) {
                const c = v >= 0 ? 'pnl-pos' : 'pnl-neg';
                const sign = v >= 0 ? '+' : '';
                html += `<div class="monthly-card"><div class="month">${m}</div>
                         <div class="val ${c}">${sign}${v}</div></div>`;
            }
            html += '</div></div>';
        }

        // Trade log table
        html += `
        <div class="chart-container">
            <h2>Trade Log (${s.trades.length} trades)</h2>
            <div style="overflow-x: auto;">
            <table>
                <thead>
                    <tr>
                        <th>Date</th><th>#</th><th>Dir</th><th>Bar</th>
                        <th>Entry</th><th>Exit</th><th>P&L</th>
                        <th>MFE</th><th>TP1</th><th>TP2</th>
                        <th>Entry Slip</th><th>Exit Slip</th><th>Total Slip</th>
                        <th>Time</th>
                    </tr>
                </thead>
                <tbody>`;

        for (const t of s.trades.slice().reverse()) {
            const pnl = parseFloat(t.pnl_pts) || 0;
            const pc = pnl >= 0 ? 'pnl-pos' : 'pnl-neg';
            const ps = pnl >= 0 ? '+' : '';
            const dir = t.direction === 'LONG' ? 'badge-long' : 'badge-short';
            const es = parseFloat(t.entry_slippage) || 0;
            const xs = parseFloat(t.exit_slippage) || 0;
            const ts = parseFloat(t.slippage_total) || 0;
            const sc = (v) => Math.abs(v) > 1 ? 'slip-bad' : 'slip-ok';
            const tp1 = t.tp1_filled ? '<span class="badge badge-tp">Y</span>' : '-';
            const tp2 = t.tp2_filled ? '<span class="badge badge-tp">Y</span>' : '-';
            const barNum = t.signal_bar || '4';
            const barRule = t.bar5_rule ? ` (${t.bar5_rule})` : '';

            html += `<tr>
                <td>${t.date}</td>
                <td>${t.trade_num}</td>
                <td><span class="badge ${dir}">${t.direction}</span></td>
                <td title="${t.bar5_rule || 'default'}">${barNum}${barRule}</td>
                <td>${t.entry}</td>
                <td>${t.exit}</td>
                <td class="${pc}">${ps}${pnl}</td>
                <td>${t.mfe}</td>
                <td>${tp1}</td><td>${tp2}</td>
                <td class="${sc(es)}">${es}</td>
                <td class="${sc(xs)}">${xs}</td>
                <td class="${sc(ts)}">${ts > 0 ? '+' : ''}${ts}</td>
                <td>${t.entry_time}–${t.exit_time}</td>
            </tr>`;
        }

        html += '</tbody></table></div></div>';
        document.getElementById('content').innerHTML = html;

        // Render charts
        renderEquity(s.equity_curve);
        renderDrawdown(s.drawdown_curve);
        renderDaily(s.daily_pnl);
        renderSlippage(s.trades);
    }

    function renderEquity(data) {
        new Chart(document.getElementById('equityChart'), {
            type: 'line',
            data: {
                labels: data.map(d => d.date),
                datasets: [{
                    label: 'Equity (pts)',
                    data: data.map(d => d.equity),
                    borderColor: '#3fb950',
                    backgroundColor: 'rgba(63,185,80,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                }]
            },
            options: chartOpts('Equity (pts)')
        });
    }

    function renderDrawdown(data) {
        new Chart(document.getElementById('ddChart'), {
            type: 'line',
            data: {
                labels: data.map(d => d.date),
                datasets: [{
                    label: 'Drawdown (pts)',
                    data: data.map(d => d.drawdown),
                    borderColor: '#f85149',
                    backgroundColor: 'rgba(248,81,73,0.1)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 3,
                }]
            },
            options: chartOpts('Drawdown (pts)')
        });
    }

    function renderDaily(data) {
        const dates = Object.keys(data);
        const vals = Object.values(data);
        new Chart(document.getElementById('dailyChart'), {
            type: 'bar',
            data: {
                labels: dates,
                datasets: [{
                    label: 'Daily P&L',
                    data: vals,
                    backgroundColor: vals.map(v => v >= 0 ? '#3fb950' : '#f85149'),
                }]
            },
            options: chartOpts('P&L (pts)')
        });
    }

    function renderSlippage(trades) {
        new Chart(document.getElementById('slipChart'), {
            type: 'bar',
            data: {
                labels: trades.map((t,i) => `#${i+1}`),
                datasets: [{
                    label: 'Slippage (pts)',
                    data: trades.map(t => parseFloat(t.slippage_total) || 0),
                    backgroundColor: trades.map(t => {
                        const s = parseFloat(t.slippage_total) || 0;
                        return s > 1 ? '#f85149' : s > 0 ? '#d29922' : '#3fb950';
                    }),
                }]
            },
            options: chartOpts('Slippage (pts)')
        });
    }

    function chartOpts(yLabel) {
        return {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#8b949e', maxRotation: 45 }, grid: { color: '#21262d' } },
                y: { ticks: { color: '#8b949e' }, grid: { color: '#21262d' },
                     title: { display: true, text: yLabel, color: '#8b949e' } }
            }
        };
    }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    mode = "DEMO" if config.IG_DEMO else "LIVE"
    return render_template_string(
        DASHBOARD_HTML,
        mode=mode,
        num_contracts=config.NUM_CONTRACTS,
        tp1=int(config.TP1_PTS),
        tp2=int(config.TP2_PTS),
    )


@app.route("/api/stats")
def api_stats():
    return jsonify(get_stats())


@app.route("/api/trades")
def api_trades():
    return jsonify(load_all_trades())


@app.route("/health")
def health():
    """Health endpoint for uptime monitors (UptimeRobot, Healthchecks.io)."""
    try:
        from shared.monitoring import get_health
        return jsonify(get_health())
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


def start_dashboard(port=8080):
    """Start the dashboard in a background thread."""
    def _run():
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=_run, daemon=True, name="dashboard")
    thread.start()
    logger.info(f"Dashboard running on http://0.0.0.0:{port}")
