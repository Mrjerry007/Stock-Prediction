const API_BASE = "http://127.0.0.1:8000";

function fmt(n) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  return Number(n).toLocaleString(undefined, { maximumFractionDigits: 2 });
}

async function loadStock() {
  const ticker = document.getElementById("ticker").value.trim().toUpperCase() || "AAPL";

  const [stockRes, predRes, alertsRes] = await Promise.all([
    fetch(`${API_BASE}/stock/${ticker}`),
    fetch(`${API_BASE}/predict/${ticker}`),
    fetch(`${API_BASE}/alerts/${ticker}`),
  ]);

  if (!stockRes.ok) throw new Error(`Stock endpoint failed for ${ticker}`);
  if (!predRes.ok) throw new Error(`Prediction endpoint failed for ${ticker}`);
  if (!alertsRes.ok) throw new Error(`Alerts endpoint failed for ${ticker}`);

  const stock = await stockRes.json();
  const pred = await predRes.json();
  const alerts = await alertsRes.json();

  const rows = stock.data || [];
  const dates = rows.map(d => d.Date);

  const candles = {
    x: dates,
    open: rows.map(d => d.Open),
    high: rows.map(d => d.High),
    low: rows.map(d => d.Low),
    close: rows.map(d => d.Close),
    type: "candlestick",
    name: "Price",
  };

  const sma20 = {
    x: dates,
    y: rows.map(d => d.SMA_20),
    type: "scatter",
    mode: "lines",
    name: "SMA 20",
  };

  const ema20 = {
    x: dates,
    y: rows.map(d => d.EMA_20),
    type: "scatter",
    mode: "lines",
    name: "EMA 20",
  };

  const macd = {
    x: dates.slice(-120),
    y: rows.slice(-120).map(d => d.MACD),
    type: "scatter",
    mode: "lines",
    name: "MACD",
    yaxis: "y2",
  };

  const layout = {
    title: `${ticker} Stock Dashboard`,
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: "#edf2ff" },
    xaxis: { rangeslider: { visible: false } },
    yaxis: { title: "Price" },
    yaxis2: {
      title: "MACD",
      overlaying: "y",
      side: "right",
      showgrid: false,
    },
    legend: { orientation: "h" },
    margin: { t: 40, r: 30, l: 50, b: 40 },
  };

  Plotly.newPlot("chart", [candles, sma20, ema20, macd], layout, { responsive: true });

  document.getElementById("currentPrice").textContent = fmt(pred.current_price);
  document.getElementById("predictedPrice").textContent = fmt(pred.predicted_next_close);
  document.getElementById("direction").textContent = pred.direction;
  document.getElementById("confidence").textContent = `${fmt(pred.confidence * 100)}%`;
  document.getElementById("modelUsed").textContent = `Model: ${pred.model_used}`;

  document.getElementById("interval").innerHTML = `
    <div><strong>Low:</strong> ${fmt(pred.interval_low)}</div>
    <div><strong>High:</strong> ${fmt(pred.interval_high)}</div>
    <div><strong>Expected Return:</strong> ${fmt(pred.predicted_return_pct)}%</div>
    <div class="meta">Prediction intervals are approximate and based on residual or volatility estimates.</div>
  `;

  const alertsBox = document.getElementById("alerts");
  alertsBox.innerHTML = "";

  if (!alerts.alerts.length) {
    alertsBox.innerHTML = `<div class="alert low">No major alerts right now.</div>`;
  } else {
    alerts.alerts.forEach(alert => {
      const el = document.createElement("div");
      el.className = `alert ${alert.severity || "low"}`;
      el.innerHTML = `
        <div><strong>${alert.type.replaceAll("_", " ").toUpperCase()}</strong></div>
        <div>${alert.message}</div>
      `;
      alertsBox.appendChild(el);
    });
  }
}

document.getElementById("loadBtn").addEventListener("click", () => {
  loadStock().catch(err => alert(err.message));
});

loadStock().catch(err => console.error(err));

setInterval(() => {
  loadStock().catch(() => {});
}, 60000);
