// ============================================
// KRAKEN SENTINEL V15.0 - DASHBOARD LOGIC
// ============================================

import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js';
import { 
    getFirestore, 
    collection, 
    query, 
    onSnapshot, 
    orderBy, 
    limit,
    where,
    getDocs
} from 'https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js';

// ============================================
// FIREBASE INITIALIZATION
// ============================================

const firebaseConfig = {
    apiKey: "YOUR_API_KEY",
    authDomain: "YOUR_PROJECT.firebaseapp.com",
    projectId: "YOUR_PROJECT_ID",
    storageBucket: "YOUR_PROJECT.appspot.com",
    messagingSenderId: "YOUR_SENDER_ID",
    appId: "YOUR_APP_ID"
};

let app, db;
let equityChart, evolutionChart;
let charts = {};

// Global state
const state = {
    equityHistory: [],
    openPositions: [],
    tradeHistory: [],
    signals: [],
    assetScores: [],
    regimeHistory: [],
    evolutionHistory: [],
    circuitBreakers: {}
};

// ============================================
// INITIALIZATION
// ============================================

async function initializeFirebase() {
    try {
        app = initializeApp(firebaseConfig);
        db = getFirestore(app);
        console.log('✅ Firebase initialized');
        setFirebaseStatus(true);
        setupAllListeners();
    } catch (error) {
        console.error('❌ Firebase init error:', error);
        setFirebaseStatus(false);
    }
}

function setFirebaseStatus(connected) {
    const statusEl = document.getElementById('firebaseStatus');
    const footerEl = document.getElementById('footerStatus');
    
    if (connected) {
        statusEl.textContent = '🟢 Connected';
        statusEl.style.color = '#00ff00';
        footerEl.textContent = 'Real-time Firestore updates active';
    } else {
        statusEl.textContent = '🔴 Disconnected';
        statusEl.style.color = '#ff0055';
        footerEl.textContent = 'Firestore connection lost. Retrying...';
    }
}

// ============================================
// CHART INITIALIZATION
// ============================================

function initCharts() {
    // Equity Chart
    const equityCtx = document.getElementById('equityChart')?.getContext('2d');
    if (equityCtx) {
        equityChart = new Chart(equityCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Equity (USD)',
                    data: [],
                    borderColor: '#00ff00',
                    backgroundColor: 'rgba(0, 255, 0, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#00ff00',
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    pointBorderColor: '#0ff',
                    pointBorderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(10, 14, 39, 0.95)',
                        titleColor: '#00ff00',
                        bodyColor: '#0ff',
                        borderColor: '#00ff00',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: false,
                        callbacks: {
                            label: (context) => `Equity: $${context.parsed.y.toFixed(2)}`
                        }
                    }
                },
                scales: {
                    y: {
                        ticks: {
                            color: '#00ff00',
                            font: { size: 11 },
                            callback: (value) => `$${value.toFixed(0)}`
                        },
                        grid: {
                            color: 'rgba(0, 255, 0, 0.1)',
                            drawBorder: false
                        }
                    },
                    x: {
                        ticks: {
                            color: '#0ff',
                            font: { size: 10 },
                            maxTicksLimit: 8
                        },
                        grid: {
                            color: 'rgba(0, 255, 255, 0.05)',
                            drawBorder: false
                        }
                    }
                }
            }
        });
    }

    // Evolution Chart
    const evolutionCtx = document.getElementById('evolutionChart')?.getContext('2d');
    if (evolutionCtx) {
        evolutionChart = new Chart(evolutionCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Best Fitness',
                    data: [],
                    borderColor: '#ff00ff',
                    backgroundColor: 'rgba(255, 0, 255, 0.1)',
                    borderWidth: 2,
                    tension: 0.4,
                    fill: true,
                    pointBackgroundColor: '#ff00ff',
                    pointRadius: 2,
                    pointHoverRadius: 4,
                    pointBorderColor: '#0ff',
                    pointBorderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(10, 14, 39, 0.95)',
                        titleColor: '#ff00ff',
                        bodyColor: '#0ff',
                        borderColor: '#ff00ff',
                        borderWidth: 1,
                        padding: 12,
                        displayColors: false,
                        callbacks: {
                            label: (context) => `Fitness: ${context.parsed.y.toFixed(4)}`
                        }
                    }
                },
                scales: {
                    y: {
                        ticks: {
                            color: '#ff00ff',
                            font: { size: 11 },
                            callback: (value) => value.toFixed(4)
                        },
                        grid: {
                            color: 'rgba(255, 0, 255, 0.1)',
                            drawBorder: false
                        }
                    },
                    x: {
                        ticks: {
                            color: '#0ff',
                            font: { size: 10 },
                            maxTicksLimit: 10
                        },
                        grid: {
                            color: 'rgba(0, 255, 255, 0.05)',
                            drawBorder: false
                        }
                    }
                }
            }
        });
    }
}

// ============================================
// FIRESTORE LISTENERS
// ============================================

function setupAllListeners() {
    // Equity History (7 days)
    const equityQuery = query(
        collection(db, 'equity_history'),
        orderBy('timestamp', 'desc'),
        limit(168)
    );
    onSnapshot(equityQuery, (snapshot) => {
        state.equityHistory = [];
        snapshot.docs.forEach(doc => {
            state.equityHistory.unshift({ id: doc.id, ...doc.data() });
        });
        updateEquityChart();
        updateMetrics();
    }, error => handleError('equity_history', error));

    // Open Positions
    onSnapshot(collection(db, 'open_positions'), (snapshot) => {
        state.openPositions = snapshot.docs.map(doc => ({
            id: doc.id,
            ...doc.data()
        }));
        updatePositionsTable();
        updateMetrics();
    }, error => handleError('open_positions', error));

    // Trade History (last 10)
    const tradeQuery = query(
        collection(db, 'trade_history'),
        orderBy('exit_timestamp', 'desc'),
        limit(10)
    );
    onSnapshot(tradeQuery, (snapshot) => {
        state.tradeHistory = snapshot.docs.map(doc => ({
            id: doc.id,
            ...doc.data()
        }));
        updateTradeHistoryTable();
        updateMetrics();
    }, error => handleError('trade_history', error));

    // Pending Signals (last 10)
    const signalsQuery = query(
        collection(db, 'signals'),
        orderBy('timestamp', 'desc'),
        limit(10)
    );
    onSnapshot(signalsQuery, (snapshot) => {
        state.signals = snapshot.docs.map(doc => ({
            id: doc.id,
            ...doc.data()
        }));
        updateSignalsTable();
    }, error => handleError('signals', error));

    // Asset Scores
    onSnapshot(collection(db, 'asset_scores'), (snapshot) => {
        state.assetScores = snapshot.docs.map(doc => ({
            pair: doc.id,
            ...doc.data()
        }));
        updateAssetScores();
    }, error => handleError('asset_scores', error));

    // Regime History (latest)
    const regimeQuery = query(
        collection(db, 'regime_history'),
        orderBy('timestamp', 'desc'),
        limit(1)
    );
    onSnapshot(regimeQuery, (snapshot) => {
        if (snapshot.docs.length > 0) {
            state.regimeHistory = [{
                id: snapshot.docs[0].id,
                ...snapshot.docs[0].data()
            }];
            updateRegimeDisplay();
        }
    }, error => handleError('regime_history', error));

    // Evolution History (last 50)
    const evolutionQuery = query(
        collection(db, 'evolution_history'),
        orderBy('timestamp', 'desc'),
        limit(50)
    );
    onSnapshot(evolutionQuery, (snapshot) => {
        state.evolutionHistory = [];
        snapshot.docs.forEach(doc => {
            state.evolutionHistory.unshift({ id: doc.id, ...doc.data() });
        });
        updateEvolutionChart();
    }, error => handleError('evolution_history', error));

    // Circuit Breakers
    onSnapshot(collection(db, 'circuit_breakers'), (snapshot) => {
        state.circuitBreakers = {};
        snapshot.docs.forEach(doc => {
            state.circuitBreakers[doc.id] = doc.data();
        });
    }, error => handleError('circuit_breakers', error));
}

// ============================================
// UPDATE FUNCTIONS
// ============================================

function updateEquityChart() {
    if (!equityChart || state.equityHistory.length === 0) return;

    const labels = state.equityHistory.map(d => {
        const ts = typeof d.timestamp === 'string' 
            ? new Date(d.timestamp) 
            : d.timestamp.toDate?.() || new Date();
        return ts.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    });

    const values = state.equityHistory.map(d => d.equity || 0);

    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = values;
    equityChart.update('none');
}

function updateEvolutionChart() {
    if (!evolutionChart || state.evolutionHistory.length === 0) return;

    const labels = state.evolutionHistory.map((_, i) => `Gen ${i}`);
    const values = state.evolutionHistory.map(e => e.best_fitness || 0);

    evolutionChart.data.labels = labels;
    evolutionChart.data.datasets[0].data = values;
    evolutionChart.update('none');
}

function updatePositionsTable() {
    const tbody = document.getElementById('positionsBody');
    if (!tbody) return;

    if (state.openPositions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="no-data">No open positions</td></tr>';
        return;
    }

    tbody.innerHTML = state.openPositions.map(pos => `
        <tr>
            <td class="pair">${escapeHtml(pos.pair)}</td>
            <td>$${formatNumber(pos.entry_price, 2)}</td>
            <td>${formatNumber(pos.size, 4)}</td>
            <td class="${(pos.unrealized_pnl || 0) >= 0 ? 'profit' : 'loss'}">
                $${formatNumber(pos.unrealized_pnl || 0, 2)}
            </td>
        </tr>
    `).join('');
}

function updateTradeHistoryTable() {
    const tbody = document.getElementById('tradeHistoryBody');
    if (!tbody) return;

    if (state.tradeHistory.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="no-data">No trade history</td></tr>';
        return;
    }

    tbody.innerHTML = state.tradeHistory.map(trade => {
        const entryTs = typeof trade.entry_timestamp === 'string'
            ? new Date(trade.entry_timestamp)
            : trade.entry_timestamp.toDate?.() || new Date();
        const exitTs = typeof trade.exit_timestamp === 'string'
            ? new Date(trade.exit_timestamp)
            : trade.exit_timestamp.toDate?.() || new Date();
        
        const duration = Math.round((exitTs - entryTs) / 1000 / 60);

        return `
            <tr>
                <td class="pair">${escapeHtml(trade.pair)}</td>
                <td>${escapeHtml(trade.strategy)}</td>
                <td>$${formatNumber(trade.entry_price, 2)}</td>
                <td>$${formatNumber(trade.exit_price, 2)}</td>
                <td class="${(trade.pnl || 0) >= 0 ? 'profit' : 'loss'}">
                    $${formatNumber(trade.pnl || 0, 2)}
                </td>
                <td class="${(trade.return_pct || 0) >= 0 ? 'profit' : 'loss'}">
                    ${formatNumber(trade.return_pct || 0, 2)}%
                </td>
                <td>${duration}m</td>
            </tr>
        `;
    }).join('');
}

function updateSignalsTable() {
    const tbody = document.getElementById('signalsBody');
    if (!tbody) return;

    if (state.signals.length === 0) {
        tbody.innerHTML = '<tr><td colspan="4" class="no-data">No pending signals</td></tr>';
        return;
    }

    tbody.innerHTML = state.signals.map(sig => `
        <tr>
            <td class="pair">${escapeHtml(sig.pair)}</td>
            <td>${escapeHtml(sig.strategy)}</td>
            <td>${formatNumber(sig.z_score, 2)}</td>
            <td>${formatNumber(sig.confidence, 2)}</td>
        </tr>
    `).join('');
}

function updateAssetScores() {
    const container = document.getElementById('assetScoresContainer');
    if (!container) return;

    const sorted = [...state.assetScores]
        .sort((a, b) => (b.composite_score || 0) - (a.composite_score || 0))
        .slice(0, 5);

    if (sorted.length === 0) {
        container.innerHTML = '<div class="no-data">No asset scores available</div>';
        return;
    }

    container.innerHTML = sorted.map(score => {
        const scoreVal = score.composite_score || 0;
        return `
            <div class="score-bar">
                <span class="pair">${escapeHtml(score.pair)}</span>
                <div class="bar">
                    <div class="fill" style="width: ${Math.min(scoreVal * 100, 100)}%"></div>
                </div>
                <span class="value">${formatNumber(scoreVal, 3)}</span>
            </div>
        `;
    }).join('');
}

function updateRegimeDisplay() {
    if (state.regimeHistory.length === 0) return;

    const regime = state.regimeHistory[0];
    const states = ['Trend ↗', 'Choppy ~', 'Ranging →', 'Vol-Low ⊕', 'Vol-High ⚡'];
    
    const stateEl = document.getElementById('regimeState');
    if (stateEl) {
        stateEl.textContent = `State: ${states[regime.current_state] || 'Unknown'}`;
    }

    const confEl = document.getElementById('regimeConfidence');
    if (confEl) {
        confEl.textContent = `Confidence: ${formatNumber(regime.confidence * 100, 1)}%`;
    }
}

function updateMetrics() {
    // Active positions
    const activePos = document.getElementById('activePositions');
    if (activePos) {
        activePos.textContent = state.openPositions.length;
    }

    // Total P&L
    let totalPnL = 0;
    let wins = 0;
    state.tradeHistory.forEach(trade => {
        const pnl = trade.pnl || 0;
        totalPnL += pnl;
        if (pnl > 0) wins++;
    });

    const pnlEl = document.getElementById('totalPnL');
    if (pnlEl) {
        pnlEl.textContent = `$${formatNumber(totalPnL, 2)}`;
        pnlEl.className = `metric-value ${totalPnL >= 0 ? 'profit' : 'loss'}`;
    }

    // Win rate
    const winRateEl = document.getElementById('winRate');
    if (winRateEl) {
        const winRate = state.tradeHistory.length > 0 
            ? (wins / state.tradeHistory.length) * 100 
            : 0;
        winRateEl.textContent = `${formatNumber(winRate, 1)}%`;
    }

    // Current equity
    const equityEl = document.getElementById('currentEquity');
    if (equityEl && state.equityHistory.length > 0) {
        const latest = state.equityHistory[state.equityHistory.length - 1];
        equityEl.textContent = `$${formatNumber(latest.equity || 300, 2)}`;
    }

    // Daily loss / unrealized
    const unrealizedPnL = state.openPositions.reduce((sum, pos) => {
        return sum + (pos.unrealized_pnl || 0);
    }, 0);

    const lossEl = document.getElementById('dailyLoss');
    if (lossEl) {
        lossEl.textContent = `$${formatNumber(unrealizedPnL, 2)}`;
        lossEl.className = `metric-value ${unrealizedPnL >= 0 ? 'profit' : 'loss'}`;
    }

    // Drawdown (calculate from equity history)
    const drawdownEl = document.getElementById('maxDrawdown');
    if (drawdownEl && state.equityHistory.length > 1) {
        let maxEquity = state.equityHistory[0].equity || 300;
        let maxDD = 0;
        state.equityHistory.forEach(entry => {
            maxEquity = Math.max(maxEquity, entry.equity || 300);
            const dd = ((maxEquity - (entry.equity || 300)) / maxEquity) * 100;
            maxDD = Math.max(maxDD, dd);
        });
        drawdownEl.textContent = `${formatNumber(maxDD, 1)}%`;
    }
}

// ============================================
// UTILITY FUNCTIONS
// ============================================

function formatNumber(num, decimals = 2) {
    if (typeof num !== 'number' || isNaN(num)) return '—';
    return num.toFixed(decimals);
}

function escapeHtml(str) {
    if (!str) return '—';
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function handleError(source, error) {
    console.error(`❌ Error in ${source}:`, error);
    // Could implement retry logic or user notification here
}

function updateTimestamp() {
    const el = document.getElementById('timestamp');
    if (el) {
        const now = new Date();
        el.textContent = now.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        });
    }
}

// ============================================
// INITIALIZATION ON PAGE LOAD
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Kraken Sentinel Dashboard...');
    initCharts();
    initializeFirebase();
    updateTimestamp();
    setInterval(updateTimestamp, 1000);
});

// Handle page visibility to optimize updates
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        console.log('📱 Dashboard hidden - reducing update frequency');
    } else {
        console.log('📱 Dashboard visible - resuming full updates');
    }
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('❌ Global error:', event.error);
});

// Unhandled promise rejection handler
window.addEventListener('unhandledrejection', (event) => {
    console.error('❌ Unhandled rejection:', event.reason);
});
