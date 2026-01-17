// Configuration
const CONFIG = {
    MIN_THRESHOLD: 100,  // Minimum stock level before alert
    MAX_THRESHOLD: 1000, // Maximum stock level before alert
    REORDER_POINT: 150,  // When to suggest reordering
    CRITICAL_DAYS: 7,    // Days until min stock is critical
    WARNING_DAYS: 14     // Days until min stock needs attention
};

// Material data structure
const materials = [
    {
        name: "Aluminum",
        currentStock: 450,
        minStock: 100,
        maxStock: 1000,
        avgConsumption: 45,
        unit: "kg",
        lastUpdated: new Date()
    },
    {
        name: "Copper",
        currentStock: 180,
        minStock: 100,
        maxStock: 800,
        avgConsumption: 35,
        unit: "kg",
        lastUpdated: new Date()
    },
    {
        name: "Boron 4%",
        currentStock: 85,
        minStock: 50,
        maxStock: 500,
        avgConsumption: 12,
        unit: "kg",
        lastUpdated: new Date()
    },
    {
        name: "Iron Metal (80%)",
        currentStock: 920,
        minStock: 150,
        maxStock: 1000,
        avgConsumption: 55,
        unit: "kg",
        lastUpdated: new Date()
    },
    {
        name: "Magnesium (99.90%)",
        currentStock: 240,
        minStock: 100,
        maxStock: 600,
        avgConsumption: 28,
        unit: "kg",
        lastUpdated: new Date()
    },
    {
        name: "Si Metal 98.5%",
        currentStock: 320,
        minStock: 150,
        maxStock: 800,
        avgConsumption: 42,
        unit: "kg",
        lastUpdated: new Date()
    },
    {
        name: "Tibor Rod 5:1",
        currentStock: 65,
        minStock: 40,
        maxStock: 400,
        avgConsumption: 8,
        unit: "kg",
        lastUpdated: new Date()
    }
];

// Scrap data - will be loaded from JSON
let scrapData = {
    total: { current: 517.9, predicted: 520.97, mae: 3.87 },
    defect: { current: 176.5, predicted: 165.48, mae: 6.16 },
    process: { current: 341.5, predicted: 356.75, mae: 11.87 }
};

let historicalData = null;
let predictionData = null;
let consumptionPredictions = null;

// Chart.js global styling for clearer visuals
if (typeof Chart !== 'undefined') {
    Chart.defaults.font.family = "'Segoe UI', 'Inter', 'system-ui', sans-serif";
    Chart.defaults.font.size = 12;
    Chart.defaults.color = '#0f172a';
    Chart.defaults.elements.line.borderWidth = 2.5;
    Chart.defaults.elements.point.radius = 4;
    Chart.defaults.elements.point.borderWidth = 2;
}

// Shared line chart options to keep visuals consistent
function getLineChartOptions({ yTickFormatter, yTitle, xTitle } = {}) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        layout: { padding: { top: 10, right: 12, bottom: 8, left: 8 } },
        plugins: {
            legend: {
                position: 'top',
                labels: {
                    usePointStyle: true,
                    padding: 16,
                    color: '#0f172a',
                    font: { size: 12, weight: '600' }
                }
            },
            tooltip: {
                backgroundColor: 'rgba(15, 23, 42, 0.92)',
                padding: 12,
                displayColors: true
            }
        },
        scales: {
            x: {
                grid: { color: 'rgba(148, 163, 184, 0.18)' },
                ticks: { color: '#475569', font: { size: 10 }, maxRotation: 35, minRotation: 0 },
                title: {
                    display: Boolean(xTitle),
                    text: xTitle || '',
                    color: '#0f172a',
                    font: { size: 12, weight: '600' }
                }
            },
            y: {
                beginAtZero: false,
                grid: { color: 'rgba(148, 163, 184, 0.18)' },
                ticks: {
                    color: '#475569',
                    font: { size: 10 },
                    callback: yTickFormatter || (value => value)
                },
                title: {
                    display: Boolean(yTitle),
                    text: yTitle || '',
                    color: '#0f172a',
                    font: { size: 12, weight: '600' }
                }
            }
        }
    };
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    updateDateTime();
    setInterval(updateDateTime, 60000); // Update every minute
    
    // Load data from JSON file
    loadDataFromJSON().then(() => {
        loadScrapData();
        renderMaterialCards();
        renderPredictionsTable();
        checkAlerts();
        initializeCharts();
        loadConsumptionPredictions();
    });
    
    // Simulate real-time updates every 5 seconds
    setInterval(simulateUpdates, 5000);
});

// Load data from JSON file
async function loadDataFromJSON() {
    try {
        const response = await fetch('data.json');
        const data = await response.json();
        
        historicalData = data.historical;
        predictionData = data.predictions;
        
        // Load future predictions
        if (data.future) {
            window.futureData = data.future;
        }
        
        // Update scrap data with latest values
        if (historicalData) {
            const lastIdx = historicalData.total.length - 1;
            scrapData.total.current = historicalData.total[lastIdx];
            scrapData.defect.current = historicalData.defect[lastIdx];
            scrapData.process.current = historicalData.process[lastIdx];
        }
        
        if (predictionData && predictionData.total && predictionData.total.length > 0) {
            const lastPred = predictionData.total[predictionData.total.length - 1];
            scrapData.total.predicted = lastPred.Predicted;
            scrapData.total.mae = Math.abs(lastPred.Error);
        }
        
        console.log('✅ Data loaded successfully:', data.metadata);
        console.log('   Historical months:', historicalData.labels.length);
        console.log('   Validation predictions:', predictionData.total.length);
        if (window.futureData) {
            console.log('   Future predictions:', window.futureData.total.length);
        }
    } catch (error) {
        console.warn('⚠️ Could not load data.json, using default data:', error);
    }
}

// Update date and time
function updateDateTime() {
    const now = new Date();
    const options = { 
        weekday: 'long', 
        year: 'numeric', 
        month: 'long', 
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    };
    document.getElementById('currentDate').textContent = now.toLocaleDateString('en-US', options);
}

// Load scrap data
function loadScrapData() {
    // Update summary cards
    document.getElementById('totalScrap').textContent = `${scrapData.total.current.toFixed(1)} kg`;
    document.getElementById('totalScrapChange').innerHTML = `
        <i class="fas fa-arrow-${scrapData.total.predicted > scrapData.total.current ? 'up' : 'down'}"></i>
        Predicted: ${scrapData.total.predicted.toFixed(1)} kg
    `;
    document.getElementById('totalScrapChange').className = `card-change ${scrapData.total.predicted > scrapData.total.current ? 'positive' : 'negative'}`;
    
    document.getElementById('defectScrap').textContent = `${scrapData.defect.current.toFixed(1)} kg`;
    document.getElementById('defectScrapChange').innerHTML = `
        <i class="fas fa-arrow-${scrapData.defect.predicted > scrapData.defect.current ? 'up' : 'down'}"></i>
        Predicted: ${scrapData.defect.predicted.toFixed(1)} kg
    `;
    document.getElementById('defectScrapChange').className = `card-change ${scrapData.defect.predicted > scrapData.defect.current ? 'positive' : 'negative'}`;
    
    document.getElementById('processScrap').textContent = `${scrapData.process.current.toFixed(1)} kg`;
    document.getElementById('processScrapChange').innerHTML = `
        <i class="fas fa-arrow-${scrapData.process.predicted > scrapData.process.current ? 'up' : 'down'}"></i>
        Predicted: ${scrapData.process.predicted.toFixed(1)} kg
    `;
    document.getElementById('processScrapChange').className = `card-change ${scrapData.process.predicted > scrapData.process.current ? 'positive' : 'negative'}`;
    
    document.getElementById('accuracy').textContent = '99.3%';
    document.getElementById('mae').textContent = scrapData.total.mae.toFixed(2);
}

// Calculate days until min/max stock
function calculateDaysUntil(current, target, avgConsumption) {
    if (avgConsumption === 0) return Infinity;
    return Math.abs((target - current) / avgConsumption);
}

// Get material status
function getMaterialStatus(material) {
    const stockPercentage = ((material.currentStock - material.minStock) / (material.maxStock - material.minStock)) * 100;
    
    if (material.currentStock <= material.minStock) {
        return { class: 'critical', label: 'CRITICAL' };
    } else if (material.currentStock >= material.maxStock * 0.9) {
        return { class: 'high', label: 'HIGH' };
    } else if (stockPercentage < 30) {
        return { class: 'low', label: 'LOW' };
    } else {
        return { class: 'ok', label: 'NORMAL' };
    }
}

// Render material cards
function renderMaterialCards() {
    const grid = document.getElementById('materialsGrid');
    grid.innerHTML = '';
    
    materials.forEach(material => {
        const daysUntilMin = calculateDaysUntil(material.currentStock, material.minStock, material.avgConsumption);
        const daysUntilMax = calculateDaysUntil(material.currentStock, material.maxStock, material.avgConsumption);
        const status = getMaterialStatus(material);
        const stockPercentage = ((material.currentStock - material.minStock) / (material.maxStock - material.minStock)) * 100;
        
        const card = document.createElement('div');
        card.className = 'material-card';
        card.innerHTML = `
            <div class="material-header">
                <div class="material-name">${material.name}</div>
                <div class="material-status status-${status.class}">${status.label}</div>
            </div>
            <div class="material-info">
                <div class="info-row">
                    <span class="info-label">Current Stock:</span>
                    <span class="info-value">${material.currentStock} ${material.unit}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Min / Max:</span>
                    <span class="info-value">${material.minStock} / ${material.maxStock} ${material.unit}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Avg Consumption:</span>
                    <span class="info-value">${material.avgConsumption} ${material.unit}/day</span>
                </div>
            </div>
            <div class="progress-bar">
                <div class="progress-fill ${stockPercentage < 30 ? 'low' : ''}" style="width: ${Math.max(0, Math.min(100, stockPercentage))}%"></div>
            </div>
            <div class="material-prediction">
                <div class="prediction-label">
                    ${material.currentStock <= material.minStock ? 'IMMEDIATE ACTION NEEDED' : 
                      daysUntilMin < CONFIG.CRITICAL_DAYS ? `⚠️ Days until minimum: ${Math.floor(daysUntilMin)}` :
                      material.currentStock >= material.maxStock * 0.9 ? `🔴 Near maximum capacity` :
                      `✓ Days until minimum: ${Math.floor(daysUntilMin)}`}
                </div>
                ${material.currentStock > material.minStock && material.currentStock < material.maxStock * 0.9 ? `
                <div class="prediction-value">
                    Reorder in ${Math.floor(calculateDaysUntil(material.currentStock, CONFIG.REORDER_POINT, material.avgConsumption))} days
                </div>
                ` : ''}
            </div>
        `;
        
        grid.appendChild(card);
    });
}

// Check and display alerts
function checkAlerts() {
    const alertsSection = document.getElementById('alertsSection');
    alertsSection.innerHTML = '';
    
    const alerts = [];
    
    materials.forEach(material => {
        const daysUntilMin = calculateDaysUntil(material.currentStock, material.minStock, material.avgConsumption);
        
        // Critical: At or below minimum
        if (material.currentStock <= material.minStock) {
            alerts.push({
                type: 'critical',
                icon: 'fa-exclamation-circle',
                title: `CRITICAL: ${material.name} Stock Depleted`,
                message: `Current stock: ${material.currentStock} ${material.unit}. IMMEDIATE ORDER REQUIRED!`,
                action: 'order',
                material: material
            });
        }
        // Critical: Will reach minimum in less than 7 days
        else if (daysUntilMin < CONFIG.CRITICAL_DAYS) {
            alerts.push({
                type: 'critical',
                icon: 'fa-exclamation-triangle',
                title: `URGENT: ${material.name} Low Stock`,
                message: `Will reach minimum in ${Math.floor(daysUntilMin)} days. Place order immediately!`,
                action: 'order',
                material: material
            });
        }
        // Warning: Will reach minimum in less than 14 days
        else if (daysUntilMin < CONFIG.WARNING_DAYS) {
            alerts.push({
                type: 'warning',
                icon: 'fa-exclamation',
                title: `Warning: ${material.name} Stock Low`,
                message: `Will reach minimum in ${Math.floor(daysUntilMin)} days. Consider placing order soon.`,
                action: 'order',
                material: material
            });
        }
        
        // At or above maximum
        if (material.currentStock >= material.maxStock) {
            alerts.push({
                type: 'info',
                icon: 'fa-info-circle',
                title: `${material.name} at Maximum Capacity`,
                message: `Current stock: ${material.currentStock} ${material.unit}. STOP ordering until stock decreases.`,
                action: 'stop',
                material: material
            });
        }
        // Near maximum (90%+)
        else if (material.currentStock >= material.maxStock * 0.9) {
            alerts.push({
                type: 'info',
                icon: 'fa-info-circle',
                title: `${material.name} Near Maximum`,
                message: `Stock at ${Math.round((material.currentStock / material.maxStock) * 100)}% capacity. Avoid new orders.`,
                action: 'stop',
                material: material
            });
        }
    });
    
    // Render alerts
    alerts.forEach(alert => {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${alert.type}`;
        alertDiv.innerHTML = `
            <div class="alert-icon">
                <i class="fas ${alert.icon}"></i>
            </div>
            <div class="alert-content">
                <div class="alert-title">${alert.title}</div>
                <div class="alert-message">${alert.message}</div>
            </div>
            <div class="alert-actions">
                ${alert.action === 'order' ? 
                    `<button class="btn btn-primary btn-sm" onclick="placeOrder('${alert.material.name}')">
                        <i class="fas fa-shopping-cart"></i> Place Order
                    </button>` :
                    `<button class="btn btn-secondary btn-sm" onclick="acknowledgeAlert('${alert.material.name}')">
                        <i class="fas fa-check"></i> Acknowledge
                    </button>`
                }
            </div>
        `;
        alertsSection.appendChild(alertDiv);
    });
}

// Render predictions table
function renderPredictionsTable() {
    const tbody = document.getElementById('predictionsTableBody');
    tbody.innerHTML = '';
    
    const currentMonth = new Date().getMonth() + 1;
    const nextMonth = currentMonth === 12 ? 1 : currentMonth + 1;
    
    materials.forEach(material => {
        const daysUntilMin = calculateDaysUntil(material.currentStock, material.minStock, material.avgConsumption);
        const status = getMaterialStatus(material);
        const predictedConsumption = material.avgConsumption * 30;
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>Month ${nextMonth}</td>
            <td>${material.name}</td>
            <td>${material.currentStock} ${material.unit}</td>
            <td>${predictedConsumption.toFixed(0)} ${material.unit}</td>
            <td>${daysUntilMin === Infinity ? 'N/A' : Math.floor(daysUntilMin)}</td>
            <td><span class="status-badge status-${status.class}">${status.label}</span></td>
            <td>
                ${daysUntilMin < CONFIG.WARNING_DAYS || material.currentStock <= material.minStock ?
                    `<button class="btn btn-primary btn-sm" onclick="placeOrder('${material.name}')">Order Now</button>` :
                    material.currentStock >= material.maxStock * 0.9 ?
                    `<span style="color: var(--danger-color);">Stop Orders</span>` :
                    `<span style="color: var(--success-color);">No Action</span>`
                }
            </td>
        `;
        tbody.appendChild(row);
    });
}

// Initialize charts
function initializeCharts() {
    // Scrap Trend Chart - Use all historical data + predictions + future
    const scrapCtx = document.getElementById('scrapTrendChart').getContext('2d');
    
    let labels = [];
    let actualData = [];
    let predictedData = [];
    
    if (historicalData && historicalData.labels) {
        // Start with all historical data
        labels = [...historicalData.labels];
        actualData = [...historicalData.total];
        
        // Initialize predicted data array with nulls
        predictedData = new Array(actualData.length).fill(null);
        
        // Find where predictions start in historical data
        let firstPredictionIdx = -1;
        
        // Add validation predictions (overlaid on last months of historical data)
        if (predictionData && predictionData.total && predictionData.total.length > 0) {
            // Sort predictions by year and month
            const sortedPredictions = [...predictionData.total].sort((a, b) => {
                if (a.Year !== b.Year) return a.Year - b.Year;
                return a.Month - b.Month;
            });
            
            sortedPredictions.forEach((pred) => {
                // Find matching index in historical data
                const histIdx = historicalData.labels.findIndex(label => {
                    const [year, month] = label.split('-');
                    return parseInt(year) === pred.Year && parseInt(month) === pred.Month;
                });
                
                if (histIdx >= 0) {
                    predictedData[histIdx] = pred.Predicted;
                    if (firstPredictionIdx === -1) {
                        firstPredictionIdx = histIdx;
                    }
                }
            });
            
            // Fill in the gap: connect from first prediction to last prediction continuously
            if (firstPredictionIdx > 0) {
                // Add a connection point from the last actual to first prediction
                predictedData[firstPredictionIdx - 1] = actualData[firstPredictionIdx - 1];
            }
        }
        
        // Add future predictions (beyond historical data) - make continuous
        if (window.futureData && window.futureData.total && window.futureData.total.length > 0) {
            // Sort future predictions
            const sortedFuture = [...window.futureData.total].sort((a, b) => {
                if (a.Year !== b.Year) return a.Year - b.Year;
                return a.Month - b.Month;
            });
            
            sortedFuture.forEach((future) => {
                const futureLabel = `${future.Year}-${String(future.Month).padStart(2, '0')}`;
                labels.push(futureLabel);
                actualData.push(null); // No actual data for future
                predictedData.push(future.Predicted);
            });
        }
    } else {
        // Fallback to default data
        labels = ['Month 9', 'Month 10', 'Month 11', 'Month 12', 'Predicted M1'];
        actualData = [691.4, 533.7, 523.1, 517.9, null];
        predictedData = [null, null, 527.77, 520.97, 518.5];
    }
    
    // Calculate and display accuracy
    let mae = 0, mape = 0, count = 0;
    if (predictionData && predictionData.total) {
        predictionData.total.forEach(pred => {
            if (pred.Error !== undefined) {
                mae += Math.abs(pred.Error);
                mape += pred.Abs_Pct_Error;
                count++;
            }
        });
        if (count > 0) {
            mae = mae / count;
            mape = mape / count;
            const accuracyDiv = document.getElementById('chartAccuracy');
            if (accuracyDiv) {
                accuracyDiv.innerHTML = `<span style="color: #10b981;">✓</span> Accuracy: MAE = ${mae.toFixed(2)} kg | MAPE = ${mape.toFixed(2)}%`;
            }
        }
    }
    
    const scrapChartOptions = getLineChartOptions({
        yTickFormatter: (value) => `${value?.toFixed ? value.toFixed(1) : value} kg`,
        yTitle: 'Scrap (kg)',
        xTitle: 'Time Period'
    });
    
    scrapChartOptions.plugins.tooltip.callbacks = {
        label: function(context) {
            let label = context.dataset.label || '';
            if (label) {
                label += ': ';
            }
            if (context.parsed.y !== null) {
                label += context.parsed.y.toFixed(1) + ' kg';
            }
            return label;
        }
    };
    
    new Chart(scrapCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Total Scrap (Actual)',
                data: actualData,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.08)',
                tension: 0.35,
                fill: true,
                pointRadius: 3,
                pointHoverRadius: 6
            }, {
                label: 'Total Scrap (Predicted)',
                data: predictedData,
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.08)',
                borderDash: [6, 4],
                tension: 0.35,
                fill: true,
                pointRadius: 3,
                pointHoverRadius: 6
            }]
        },
        options: scrapChartOptions
    });
}

// Place order function
function placeOrder(materialName) {
    const material = materials.find(m => m.name === materialName);
    if (!material) return;
    
    const orderQuantity = material.maxStock - material.currentStock;
    
    document.getElementById('modalTitle').innerHTML = '<i class="fas fa-shopping-cart"></i> Place Order';
    document.getElementById('modalBody').innerHTML = `
        <h3>${material.name}</h3>
        <div style="margin: 20px 0;">
            <p><strong>Current Stock:</strong> ${material.currentStock} ${material.unit}</p>
            <p><strong>Minimum Stock:</strong> ${material.minStock} ${material.unit}</p>
            <p><strong>Maximum Stock:</strong> ${material.maxStock} ${material.unit}</p>
            <p><strong>Suggested Order Quantity:</strong> ${orderQuantity} ${material.unit}</p>
            <p style="color: var(--warning-color); margin-top: 20px;">
                <i class="fas fa-exclamation-triangle"></i>
                This will bring stock to maximum capacity.
            </p>
        </div>
        <div style="margin-top: 20px;">
            <label style="display: block; margin-bottom: 10px; font-weight: 600;">Order Quantity (${material.unit}):</label>
            <input type="number" id="orderQuantity" value="${orderQuantity}" min="1" 
                   style="width: 100%; padding: 10px; border: 1px solid var(--border-color); border-radius: 6px;">
        </div>
    `;
    
    document.getElementById('alertModal').classList.add('active');
}

// Acknowledge alert
function acknowledgeAlert(materialName) {
    alert(`Alert acknowledged for ${materialName || 'item'}`);
}

// Close modal
function closeModal() {
    document.getElementById('alertModal').classList.remove('active');
}

// Refresh data
function refreshData() {
    const btn = document.querySelector('.btn-refresh i');
    btn.style.transform = 'rotate(360deg)';
    
    setTimeout(() => {
        btn.style.transform = 'rotate(0deg)';
        loadScrapData();
        renderMaterialCards();
        renderPredictionsTable();
        checkAlerts();
    }, 500);
}

// Simulate real-time updates
function simulateUpdates() {
    // Randomly update material stocks (simulate consumption)
    materials.forEach(material => {
        const randomConsumption = Math.random() * material.avgConsumption * 0.1;
        material.currentStock = Math.max(0, material.currentStock - randomConsumption);
    });
    
    // Update displays
    renderMaterialCards();
    checkAlerts();
}

// Load consumption predictions
async function loadConsumptionPredictions() {
    try {
        // Try local path first
        const response = await fetch('./consumption_predictions.json');
        if (!response.ok) throw new Error('Failed to fetch');
        
        consumptionPredictions = await response.json();
        
        console.log('✅ Consumption predictions loaded', consumptionPredictions);
        console.log('Materials found:', Object.keys(consumptionPredictions));
        
        // Render consumption charts
        renderIndividualMaterialCharts();
        renderConsumptionCharts();
    } catch (error) {
        console.error('❌ Error loading consumption predictions:', error);
    }
}

// Render consumption prediction charts
function renderConsumptionCharts() {
    if (!consumptionPredictions) {
        console.warn('No consumption predictions for main chart');
        return;
    }
    
    // Get first material for main chart
    const firstMaterial = Object.keys(consumptionPredictions)[0];
    if (!firstMaterial) {
        console.warn('No materials found');
        return;
    }
    
    const data = consumptionPredictions[firstMaterial];
    
    console.log(`Creating main consumption chart for ${firstMaterial}`);
    
    const canvasEl = document.getElementById('consumptionPredChart');
    if (!canvasEl) {
        console.error('consumptionPredChart canvas not found');
        return;
    }
    
    const ctx = canvasEl.getContext('2d');
    const mainConsumptionOptions = getLineChartOptions({
        yTickFormatter: (value) => `${value?.toFixed ? value.toFixed(1) : value} kg`,
        yTitle: 'Consumption (kg)',
        xTitle: 'Time Period'
    });
    
    mainConsumptionOptions.plugins.tooltip.callbacks = {
        label: function(context) {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            if (value === null || value === undefined) return label;
            return `${label}: ${value.toFixed(2)} kg`;
        }
    };
    
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [{
                label: 'Actual Consumption',
                data: data.actuals,
                borderColor: '#2563eb',
                backgroundColor: 'rgba(37, 99, 235, 0.08)',
                borderWidth: 3,
                tension: 0.35,
                fill: true,
                pointRadius: 4,
                pointBackgroundColor: '#2563eb',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }, {
                label: 'Predicted Consumption',
                data: data.predictions,
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.08)',
                borderDash: [8, 4],
                borderWidth: 3,
                tension: 0.35,
                fill: true,
                pointRadius: 4,
                pointBackgroundColor: '#f59e0b',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: mainConsumptionOptions
    });
    
    // Update accuracy display
    const accuracyColor = data.mape < 5 ? '#10b981' : data.mape < 10 ? '#f59e0b' : '#ef4444';
    document.getElementById('consumptionAccuracy').innerHTML = 
        `<span style="color: ${accuracyColor};">✓</span> ${firstMaterial}: MAE = ${data.mae.toFixed(2)} kg | MAPE = ${data.mape.toFixed(2)}%`;
}

// Render individual material charts
function renderIndividualMaterialCharts() {
    if (!consumptionPredictions) {
        console.warn('No consumption predictions loaded');
        return;
    }
    
    console.log('Rendering individual charts for materials:', Object.keys(consumptionPredictions));
    
    const materialMapping = {
        'Boron 4%': { id: 'boron', accuracyId: 'boron_accuracy' },
        'Iron Metal (80%)': { id: 'iron', accuracyId: 'iron_accuracy' },
        'Magnesium (99.90%)': { id: 'magnesium', accuracyId: 'magnesium_accuracy' },
        'Si Metal 98.5%': { id: 'si', accuracyId: 'si_accuracy' },
        'Tibor Rod 5:1': { id: 'tibor', accuracyId: 'tibor_accuracy' }
    };
    
    Object.keys(consumptionPredictions).forEach(materialName => {
        const data = consumptionPredictions[materialName];
        const mapping = materialMapping[materialName];
        
        if (!mapping) {
            console.warn(`No mapping for material: ${materialName}`);
            return;
        }
        
        console.log(`Rendering chart for ${materialName}`);
        
        // Update accuracy display
        const accuracyColor = data.mape < 5 ? '#10b981' : data.mape < 10 ? '#f59e0b' : '#ef4444';
        const accuracyEl = document.getElementById(mapping.accuracyId);
        if (accuracyEl) {
            accuracyEl.innerHTML = `<span style="color: ${accuracyColor};">✓</span> MAE: ${data.mae.toFixed(2)} kg | MAPE: ${data.mape.toFixed(2)}%`;
            accuracyEl.style.color = accuracyColor;
        }
        
        // Get canvas element
        const canvasId = `${mapping.id}_chart`;
        const canvasEl = document.getElementById(canvasId);
        
        if (!canvasEl) {
            console.error(`Canvas element not found: ${canvasId}`);
            return;
        }
        
        // Destroy existing chart if it exists (Chart.js v3+)
        const existingChart = Chart.getChart(canvasEl);
        if (existingChart) {
            existingChart.destroy();
        }
        
        console.log(`Creating chart for ${materialName} with ${data.dates.length} data points`);
        
        // Create chart
        const ctx = canvasEl.getContext('2d');
        const individualChartOptions = getLineChartOptions({
            yTickFormatter: (value) => `${value?.toFixed ? value.toFixed(1) : value} kg`,
            yTitle: 'Consumption (kg)'
        });
        
        individualChartOptions.plugins.tooltip.callbacks = {
            title: function(context) {
                return 'Date: ' + context[0].label;
            },
            label: function(context) {
                const value = context.parsed.y;
                const label = context.dataset.label;
                return `${label}: ${value.toFixed(2)} kg`;
            },
            afterLabel: function(context) {
                if (context.datasetIndex === 1) {
                    const actual = data.actuals[context.dataIndex];
                    const predicted = data.predictions[context.dataIndex];
                    const error = Math.abs(actual - predicted);
                    const errorPct = ((error / actual) * 100).toFixed(2);
                    return 'Error: ' + error.toFixed(2) + ' kg (' + errorPct + '%)';
                }
                return '';
            }
        };
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.dates,
                datasets: [{
                    label: 'Actual Consumption',
                    data: data.actuals,
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.08)',
                    borderWidth: 3,
                    tension: 0.35,
                    fill: true,
                    pointRadius: 4,
                    pointBackgroundColor: '#2563eb',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointHoverRadius: 7
                }, {
                    label: 'Predicted Consumption',
                    data: data.predictions,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.08)',
                    borderWidth: 3,
                    borderDash: [8, 4],
                    tension: 0.35,
                    fill: true,
                    pointRadius: 4,
                    pointBackgroundColor: '#f59e0b',
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointHoverRadius: 7
                }]
            },
            options: individualChartOptions
        });
        
        console.log(`✅ Chart created for ${materialName}`);
    });
}

// Render consumption accuracy cards (removed - using individual charts instead)
function renderConsumptionAccuracyCards() {
    // This function is no longer needed as accuracy is shown in individual charts
}
