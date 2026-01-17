# Arabcap Casting Material Inventory Dashboard

## Features

### 1. Real-Time Material Monitoring
- Displays current stock levels for all materials
- Shows consumption rates and predictions
- Visual progress bars for stock levels
- Color-coded status indicators (Normal, Low, Critical, High)

### 2. Intelligent Alerting System

#### Low Stock Alerts
- **Critical Alert**: When stock reaches or falls below minimum threshold
  - Immediate action required
  - "Place Order" button for quick ordering
  
- **Warning Alert**: When stock will reach minimum in < 14 days
  - Suggests placing order soon
  - Shows days until minimum reached

#### High Stock Alerts
- **Info Alert**: When stock reaches or exceeds maximum threshold
  - Stops new orders to prevent overstocking
  - Shows percentage of capacity used

### 3. Prediction Features
- **Days Until Minimum**: Calculates when material will reach minimum stock based on consumption rate
- **Days Until Maximum**: Calculates when material will reach maximum capacity
- **Reorder Point**: Suggests optimal time to reorder
- **Consumption Forecasting**: Predicts future material usage

### 4. Visual Dashboard Components
- **Summary Cards**: Quick overview of scrap predictions and accuracy
- **Material Cards**: Individual cards for each material with detailed info
- **Charts**: 
  - Scrap trend predictions (actual vs predicted)
  - Material consumption forecast (bar chart)
- **Predictions Table**: Detailed table view of all materials and their status

### 5. Action Buttons
- **Place Order**: Quick order button for low-stock materials
- **Acknowledge**: Acknowledge alerts and warnings
- **Refresh**: Manual data refresh option

## Configuration

Edit the `CONFIG` object in `app.js` to adjust thresholds:

```javascript
const CONFIG = {
    MIN_THRESHOLD: 100,  // Minimum stock level before alert
    MAX_THRESHOLD: 1000, // Maximum stock level before alert
    REORDER_POINT: 150,  // When to suggest reordering
    CRITICAL_DAYS: 7,    // Days until min stock is critical
    WARNING_DAYS: 14     // Days until min stock needs attention
};
```

## How to Use

### Option 1: Standalone (Static Dashboard)
1. Simply open `index.html` in a web browser
2. The dashboard will display with simulated data

### Option 2: With API Backend
1. Install Flask and dependencies:
   ```bash
   pip install flask flask-cors pandas numpy
   ```

2. Start the API server:
   ```bash
   cd api
   python3 predictions.py
   ```

3. Update `app.js` to fetch from API:
   ```javascript
   // Add at the top of app.js
   const API_URL = 'http://localhost:5000/api';
   
   // Use fetch to get data
   fetch(`${API_URL}/summary`)
       .then(response => response.json())
       .then(data => updateDashboard(data));
   ```

4. Open `index.html` in a web browser

## Alert Logic

### When to Order (Low Stock)
1. **Immediate**: Stock ≤ Minimum Threshold
2. **Critical**: Will reach minimum in < 7 days
3. **Warning**: Will reach minimum in < 14 days

### When to Stop Ordering (High Stock)
1. **Maximum**: Stock ≥ Maximum Threshold
2. **Near Maximum**: Stock ≥ 90% of Maximum Threshold

## Calculation Formulas

### Days Until Minimum/Maximum
```
days = |target_stock - current_stock| / avg_daily_consumption
```

### Stock Status
```
percentage = (current_stock - min_stock) / (max_stock - min_stock) × 100
```

- **Critical**: ≤ Minimum or percentage < 0%
- **Low**: 0% < percentage < 30%
- **Normal**: 30% ≤ percentage < 90%
- **High**: percentage ≥ 90%

## Customization

### Adding New Materials
Edit the `materials` array in `app.js`:

```javascript
{
    name: "Material Name",
    currentStock: 500,
    minStock: 100,
    maxStock: 1000,
    avgConsumption: 45,
    unit: "kg",
    lastUpdated: new Date()
}
```

### Changing Colors/Styling
Edit `styles.css` CSS variables:

```css
:root {
    --primary-color: #2563eb;
    --danger-color: #ef4444;
    --warning-color: #f59e0b;
    --success-color: #10b981;
}
```

## Browser Compatibility
- Chrome/Edge: ✅ Fully supported
- Firefox: ✅ Fully supported
- Safari: ✅ Fully supported
- Mobile browsers: ✅ Responsive design

## Future Enhancements
- Email/SMS notifications for critical alerts
- Historical data analysis and reporting
- Integration with ERP systems
- Multi-user access with authentication
- Export reports to PDF/Excel
- Machine learning for improved predictions
