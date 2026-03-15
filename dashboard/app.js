/* ============================================
   FFAC 26 Data Explorer — Dashboard Logic
   ============================================ */

let rawData = [];
let headers = [];
let numericCols = [];
let textCols = [];
let charts = {};

// ─── CSV Parsing ───────────────────────────────
function parseCSV(text) {
    const lines = text.trim().split('\n');
    const hdrs = parseCSVLine(lines[0]);
    const rows = [];
    for (let i = 1; i < lines.length; i++) {
        const vals = parseCSVLine(lines[i]);
        if (vals.length === hdrs.length) {
            const row = {};
            hdrs.forEach((h, j) => {
                const v = vals[j].trim();
                const num = Number(v);
                row[h] = v === '' ? null : (isNaN(num) ? v : num);
            });
            rows.push(row);
        }
    }
    return { headers: hdrs, rows };
}

function parseCSVLine(line) {
    const result = [];
    let current = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
            inQuotes = !inQuotes;
        } else if (ch === ',' && !inQuotes) {
            result.push(current);
            current = '';
        } else {
            current += ch;
        }
    }
    result.push(current);
    return result;
}

function detectColumnTypes(data, hdrs) {
    const numeric = [];
    const text = [];
    hdrs.forEach(h => {
        const sample = data.filter(r => r[h] !== null).slice(0, 50);
        const isNum = sample.length > 0 && sample.every(r => typeof r[h] === 'number');
        if (isNum) numeric.push(h);
        else text.push(h);
    });
    return { numeric, text };
}

// ─── File Upload ───────────────────────────────
document.getElementById('csvInput').addEventListener('change', e => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = ev => {
        const { headers: h, rows } = parseCSV(ev.target.result);
        headers = h;
        rawData = rows;
        const types = detectColumnTypes(rawData, headers);
        numericCols = types.numeric;
        textCols = types.text;

        document.getElementById('fileStatus').textContent = `${file.name} (${rawData.length} rows)`;
        document.getElementById('fileStatus').classList.add('loaded');
        document.getElementById('emptyState').style.display = 'none';

        initAllTabs();
        showTab('overview');
    };
    reader.readAsText(file);
});

// ─── Tabs ──────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        if (rawData.length === 0) return;
        showTab(tab.dataset.tab);
    });
});

function showTab(name) {
    document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === name));
    document.querySelectorAll('.tab-content').forEach(c => c.style.display = 'none');
    const el = document.getElementById(`tab-${name}`);
    if (el) el.style.display = 'block';
}

// ─── Init All Tabs ─────────────────────────────
function initAllTabs() {
    initOverview();
    initDistributions();
    initCorrelations();
    initSeeds();
    initExplorer();
}

// ─── Overview Tab ──────────────────────────────
function initOverview() {
    const missingCounts = {};
    headers.forEach(h => {
        missingCounts[h] = rawData.filter(r => r[h] === null || r[h] === '').length;
    });
    const totalMissing = Object.values(missingCounts).reduce((a, b) => a + b, 0);
    const totalCells = rawData.length * headers.length;

    // Metrics
    const grid = document.getElementById('metricsGrid');
    grid.innerHTML = [
        metricCard(rawData.length.toLocaleString(), 'Rows'),
        metricCard(headers.length, 'Columns'),
        metricCard(numericCols.length, 'Numeric'),
        metricCard(textCols.length, 'Categorical'),
        metricCard(`${((totalMissing / totalCells) * 100).toFixed(1)}%`, 'Missing'),
    ].join('');

    // Missing chart
    const colsWithMissing = headers.filter(h => missingCounts[h] > 0).sort((a, b) => missingCounts[b] - missingCounts[a]).slice(0, 20);
    destroyChart('missingChart');
    if (colsWithMissing.length > 0) {
        charts.missingChart = new Chart(document.getElementById('missingChart'), {
            type: 'bar',
            data: {
                labels: colsWithMissing,
                datasets: [{
                    label: 'Missing Count',
                    data: colsWithMissing.map(h => missingCounts[h]),
                    backgroundColor: 'rgba(108, 99, 255, 0.6)',
                    borderColor: '#6c63ff',
                    borderWidth: 1,
                    borderRadius: 4,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#9aa0b0', font: { size: 10 } }, grid: { color: '#2a2d3e' } },
                    y: { ticks: { color: '#9aa0b0' }, grid: { color: '#2a2d3e' } },
                }
            }
        });
    }

    // Column types
    const typesEl = document.getElementById('columnTypes');
    typesEl.innerHTML = headers.map(h => {
        const isNum = numericCols.includes(h);
        return `<div class="col-type-item">
            <span class="col-type-name">${h}</span>
            <span class="col-type-badge ${isNum ? 'numeric' : 'text'}">${isNum ? 'numeric' : 'text'}</span>
        </div>`;
    }).join('');

    // Sample table
    renderTable('sampleTable', rawData.slice(0, 10));
}

function metricCard(value, label) {
    return `<div class="metric-card">
        <div class="metric-value">${value}</div>
        <div class="metric-label">${label}</div>
    </div>`;
}

// ─── Distributions Tab ─────────────────────────
function initDistributions() {
    const select = document.getElementById('distColumn');
    select.innerHTML = numericCols.map(c => `<option value="${c}">${c}</option>`).join('');
    select.addEventListener('change', renderDistribution);
    document.getElementById('distType').addEventListener('change', renderDistribution);
    if (numericCols.length) renderDistribution();
}

function renderDistribution() {
    const col = document.getElementById('distColumn').value;
    const type = document.getElementById('distType').value;
    const values = rawData.map(r => r[col]).filter(v => v !== null && typeof v === 'number');

    document.getElementById('distTitle').textContent = `${col} — ${type === 'histogram' ? 'Histogram' : 'Box Plot Stats'}`;

    // Stats
    const sorted = [...values].sort((a, b) => a - b);
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const median = sorted[Math.floor(sorted.length / 2)];
    const std = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length);
    const min = sorted[0];
    const max = sorted[sorted.length - 1];
    const q1 = sorted[Math.floor(sorted.length * 0.25)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];

    document.getElementById('distStats').innerHTML = [
        statItem(values.length, 'Count'),
        statItem(mean.toFixed(2), 'Mean'),
        statItem(median.toFixed(2), 'Median'),
        statItem(std.toFixed(2), 'Std Dev'),
        statItem(min.toFixed(2), 'Min'),
        statItem(q1.toFixed(2), 'Q1'),
        statItem(q3.toFixed(2), 'Q3'),
        statItem(max.toFixed(2), 'Max'),
    ].join('');

    // Chart
    destroyChart('distChart');
    if (type === 'histogram') {
        const bins = 30;
        const binWidth = (max - min) / bins || 1;
        const counts = new Array(bins).fill(0);
        values.forEach(v => {
            const idx = Math.min(Math.floor((v - min) / binWidth), bins - 1);
            counts[idx]++;
        });
        const labels = counts.map((_, i) => (min + i * binWidth).toFixed(1));

        charts.distChart = new Chart(document.getElementById('distChart'), {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label: col,
                    data: counts,
                    backgroundColor: 'rgba(108, 99, 255, 0.5)',
                    borderColor: '#6c63ff',
                    borderWidth: 1,
                    borderRadius: 2,
                }]
            },
            options: chartDefaults()
        });
    } else {
        // Box plot as a horizontal bar showing min, Q1, median, Q3, max
        charts.distChart = new Chart(document.getElementById('distChart'), {
            type: 'bar',
            data: {
                labels: ['Min', 'Q1', 'Median', 'Q3', 'Max'],
                datasets: [{
                    label: col,
                    data: [min, q1, median, q3, max],
                    backgroundColor: [
                        'rgba(255, 107, 107, 0.6)',
                        'rgba(255, 179, 71, 0.6)',
                        'rgba(0, 212, 170, 0.6)',
                        'rgba(255, 179, 71, 0.6)',
                        'rgba(255, 107, 107, 0.6)',
                    ],
                    borderWidth: 1,
                    borderRadius: 4,
                }]
            },
            options: chartDefaults()
        });
    }
}

function statItem(value, label) {
    return `<div class="stat-item"><div class="stat-value">${value}</div><div class="stat-label">${label}</div></div>`;
}

// ─── Correlations Tab ──────────────────────────
function initCorrelations() {
    if (numericCols.length < 2) return;

    // Compute correlation matrix
    const cols = numericCols.slice(0, 20); // Cap for readability
    const matrix = [];
    const means = {};
    const stds = {};

    cols.forEach(c => {
        const vals = rawData.map(r => r[c]).filter(v => v !== null && typeof v === 'number');
        means[c] = vals.reduce((a, b) => a + b, 0) / vals.length;
        stds[c] = Math.sqrt(vals.reduce((a, b) => a + (b - means[c]) ** 2, 0) / vals.length) || 1;
    });

    for (let i = 0; i < cols.length; i++) {
        matrix[i] = [];
        for (let j = 0; j < cols.length; j++) {
            if (i === j) { matrix[i][j] = 1; continue; }
            let sum = 0, count = 0;
            rawData.forEach(r => {
                if (r[cols[i]] !== null && r[cols[j]] !== null &&
                    typeof r[cols[i]] === 'number' && typeof r[cols[j]] === 'number') {
                    sum += ((r[cols[i]] - means[cols[i]]) / stds[cols[i]]) *
                        ((r[cols[j]] - means[cols[j]]) / stds[cols[j]]);
                    count++;
                }
            });
            matrix[i][j] = count > 1 ? sum / (count - 1) : 0;
        }
    }

    // Render heatmap as scatter
    const heatData = [];
    for (let i = 0; i < cols.length; i++) {
        for (let j = 0; j < cols.length; j++) {
            heatData.push({ x: j, y: i, v: matrix[i][j] });
        }
    }

    destroyChart('corrChart');
    charts.corrChart = new Chart(document.getElementById('corrChart'), {
        type: 'scatter',
        data: {
            datasets: [{
                data: heatData.map(d => ({ x: d.x, y: d.y })),
                backgroundColor: heatData.map(d => corrColor(d.v)),
                pointRadius: Math.min(16, 200 / cols.length),
                pointStyle: 'rect',
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: ctx => {
                            const d = heatData[ctx.dataIndex];
                            return `${cols[d.y]} × ${cols[d.x]}: ${d.v.toFixed(3)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear', min: -0.5, max: cols.length - 0.5,
                    ticks: { callback: v => cols[v] || '', color: '#9aa0b0', font: { size: 9 }, maxRotation: 90 },
                    grid: { display: false }
                },
                y: {
                    type: 'linear', min: -0.5, max: cols.length - 0.5,
                    ticks: { callback: v => cols[v] || '', color: '#9aa0b0', font: { size: 9 } },
                    grid: { display: false },
                    reverse: true
                }
            }
        }
    });

    // Top correlations with target (seed)
    const targetCol = findSeedColumn();
    const topCorr = document.getElementById('topCorrelations');
    if (targetCol && numericCols.includes(targetCol)) {
        const idx = cols.indexOf(targetCol);
        if (idx >= 0) {
            const corrs = cols.map((c, i) => ({ name: c, value: matrix[idx][i] }))
                .filter(c => c.name !== targetCol)
                .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
                .slice(0, 15);

            topCorr.innerHTML = corrs.map(c => {
                const pct = Math.abs(c.value) * 100;
                const cls = c.value >= 0 ? 'positive' : 'negative';
                const color = c.value >= 0 ? '#6c63ff' : '#ff6b6b';
                return `<div class="corr-item">
                    <span class="corr-name">${c.name}</span>
                    <div class="corr-bar"><div class="corr-bar-fill" style="width:${pct}%;background:${color}"></div></div>
                    <span class="corr-value ${cls}">${c.value.toFixed(3)}</span>
                </div>`;
            }).join('');
        } else {
            topCorr.innerHTML = `<p style="color:var(--text-muted)">Target column "${targetCol}" not in top 20 numeric columns.</p>`;
        }
    } else {
        topCorr.innerHTML = `<p style="color:var(--text-muted)">No seed/target column detected for ranking.</p>`;
    }
}

function corrColor(v) {
    // Blue for positive, red for negative, intensity by magnitude
    const a = Math.min(Math.abs(v), 1);
    if (v >= 0) return `rgba(108, 99, 255, ${0.1 + a * 0.8})`;
    return `rgba(255, 107, 107, ${0.1 + a * 0.8})`;
}

// ─── Seed Analysis Tab ─────────────────────────
function initSeeds() {
    const seedCol = findSeedColumn();
    if (!seedCol) return;

    const seedValues = rawData.map(r => r[seedCol]).filter(v => v !== null && typeof v === 'number');
    const seedCounts = {};
    seedValues.forEach(s => { seedCounts[s] = (seedCounts[s] || 0) + 1; });
    const sortedSeeds = Object.keys(seedCounts).sort((a, b) => Number(a) - Number(b));

    destroyChart('seedDistChart');
    charts.seedDistChart = new Chart(document.getElementById('seedDistChart'), {
        type: 'bar',
        data: {
            labels: sortedSeeds,
            datasets: [{
                label: 'Count',
                data: sortedSeeds.map(s => seedCounts[s]),
                backgroundColor: sortedSeeds.map((_, i) =>
                    `hsl(${250 - i * (200 / sortedSeeds.length)}, 70%, 60%)`
                ),
                borderRadius: 4,
            }]
        },
        options: {
            ...chartDefaults(),
            plugins: { legend: { display: false } },
        }
    });

    // Metric selector
    const metricSelect = document.getElementById('seedMetric');
    const availableMetrics = numericCols.filter(c => c !== seedCol);
    metricSelect.innerHTML = availableMetrics.map(c => `<option value="${c}">${c}</option>`).join('');
    metricSelect.addEventListener('change', () => renderSeedMetric(seedCol));
    if (availableMetrics.length) renderSeedMetric(seedCol);
}

function renderSeedMetric(seedCol) {
    const metric = document.getElementById('seedMetric').value;
    if (!metric) return;

    // Group by seed, compute mean
    const groups = {};
    rawData.forEach(r => {
        const s = r[seedCol];
        if (s !== null && r[metric] !== null && typeof r[metric] === 'number') {
            if (!groups[s]) groups[s] = [];
            groups[s].push(r[metric]);
        }
    });

    const seeds = Object.keys(groups).sort((a, b) => Number(a) - Number(b));
    const means = seeds.map(s => groups[s].reduce((a, b) => a + b, 0) / groups[s].length);

    destroyChart('seedMetricChart');
    charts.seedMetricChart = new Chart(document.getElementById('seedMetricChart'), {
        type: 'line',
        data: {
            labels: seeds.map(s => `Seed ${s}`),
            datasets: [{
                label: `Avg ${metric}`,
                data: means,
                borderColor: '#6c63ff',
                backgroundColor: 'rgba(108, 99, 255, 0.1)',
                fill: true,
                tension: 0.3,
                pointBackgroundColor: '#6c63ff',
                pointRadius: 5,
                pointHoverRadius: 8,
            }]
        },
        options: chartDefaults()
    });
}

function findSeedColumn() {
    // Try to find the seed column by common names
    const patterns = ['overall_seed', 'overall seed', 'seed', 'tournament_seed'];
    for (const p of patterns) {
        const match = headers.find(h => h.toLowerCase().replace(/\s+/g, '_') === p || h.toLowerCase() === p);
        if (match) return match;
    }
    return null;
}

// ─── Data Explorer Tab ─────────────────────────
let explorerPage = 0;

function initExplorer() {
    explorerPage = 0;
    renderExplorer();

    document.getElementById('explorerSearch').addEventListener('input', () => {
        explorerPage = 0;
        renderExplorer();
    });
    document.getElementById('explorerPageSize').addEventListener('change', () => {
        explorerPage = 0;
        renderExplorer();
    });
    document.getElementById('exportCsv').addEventListener('click', exportFilteredCSV);
}

function getFilteredData() {
    const search = document.getElementById('explorerSearch').value.toLowerCase();
    if (!search) return rawData;
    return rawData.filter(r =>
        Object.values(r).some(v => v !== null && String(v).toLowerCase().includes(search))
    );
}

function renderExplorer() {
    const filtered = getFilteredData();
    const pageSize = parseInt(document.getElementById('explorerPageSize').value);
    const totalPages = Math.ceil(filtered.length / pageSize);
    const start = explorerPage * pageSize;
    const page = filtered.slice(start, start + pageSize);

    renderTable('explorerTable', page, true);

    // Pagination
    const pag = document.getElementById('pagination');
    let html = `<button ${explorerPage === 0 ? 'disabled' : ''} onclick="explorerPage--;renderExplorer()">← Prev</button>`;
    html += `<span style="color:var(--text-secondary);font-size:0.8rem">${start + 1}–${Math.min(start + pageSize, filtered.length)} of ${filtered.length}</span>`;
    html += `<button ${explorerPage >= totalPages - 1 ? 'disabled' : ''} onclick="explorerPage++;renderExplorer()">Next →</button>`;
    pag.innerHTML = html;
}

function exportFilteredCSV() {
    const filtered = getFilteredData();
    let csv = headers.join(',') + '\n';
    filtered.forEach(r => {
        csv += headers.map(h => {
            const v = r[h];
            if (v === null) return '';
            return typeof v === 'string' && v.includes(',') ? `"${v}"` : v;
        }).join(',') + '\n';
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'filtered_data.csv';
    a.click();
    URL.revokeObjectURL(url);
}

// ─── Table Renderer ────────────────────────────
function renderTable(id, data, sortable = false) {
    const table = document.getElementById(id);
    if (!data.length) {
        table.innerHTML = '<tr><td style="text-align:center;color:var(--text-muted)">No data</td></tr>';
        return;
    }

    const cols = headers;
    let html = '<thead><tr>';
    cols.forEach(c => {
        html += `<th>${c}</th>`;
    });
    html += '</tr></thead><tbody>';
    data.forEach(r => {
        html += '<tr>';
        cols.forEach(c => {
            const v = r[c];
            html += `<td>${v === null ? '<span style="color:var(--danger)">null</span>' : v}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody>';
    table.innerHTML = html;
}

// ─── Chart Helpers ─────────────────────────────
function destroyChart(id) {
    if (charts[id]) {
        charts[id].destroy();
        delete charts[id];
    }
}

function chartDefaults() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { color: '#9aa0b0' } }
        },
        scales: {
            x: { ticks: { color: '#9aa0b0', font: { size: 10 } }, grid: { color: '#2a2d3e' } },
            y: { ticks: { color: '#9aa0b0' }, grid: { color: '#2a2d3e' } },
        }
    };
}
