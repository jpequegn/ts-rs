# Industrial Sensor Monitoring Example

Complete workflow for analyzing industrial sensor data using Chronos for predictive maintenance and quality control.

## Business Context

**Objective**: Monitor industrial equipment using IoT sensors to:
- Predict equipment failures before they occur
- Optimize maintenance schedules and costs
- Ensure product quality and safety compliance
- Minimize unplanned downtime and production losses

**Business Value**:
- 15-30% reduction in maintenance costs
- 70% reduction in unplanned downtime
- Improved product quality and consistency
- Enhanced workplace safety
- Regulatory compliance documentation

## Dataset Description

**Data Source**: Industrial IoT sensors on manufacturing equipment
**Equipment**: Large industrial compressor system
**Time Period**: 6 months of continuous monitoring
**Frequency**: 1-minute intervals
**Sensors**:
- Temperature sensors (inlet, outlet, bearing)
- Pressure sensors (suction, discharge)
- Vibration sensors (motor, compressor)
- Flow rate sensor
- Power consumption meter

**Sample Data Format**:
```csv
timestamp,temp_inlet,temp_outlet,temp_bearing,pressure_suction,pressure_discharge,vibration_motor,vibration_compressor,flow_rate,power_consumption,maintenance_flag
2023-01-01 00:00:00,22.5,45.3,38.2,2.1,8.7,0.15,0.23,150.2,245.6,0
2023-01-01 00:01:00,22.7,45.8,38.4,2.0,8.8,0.16,0.24,149.8,247.1,0
2023-01-01 00:02:00,22.6,45.5,38.6,2.1,8.9,0.14,0.22,150.5,246.3,0
```

## Getting Sample Data

### Option 1: Generate Realistic Industrial Sensor Data
```bash
# Generate synthetic sensor data with realistic patterns
chronos import --generate \
  --output industrial_sensors.csv \
  --points 259200 \
  --frequency "1min" \
  --start-date "2023-01-01" \
  --pattern industrial \
  --sensors "temperature:3,pressure:2,vibration:2,flow:1,power:1" \
  --include-failures \
  --failure-rate 0.001
```

### Option 2: Use Provided Sample Dataset
```bash
# Download sample industrial sensor data
curl -o industrial_sensors_sample.csv \
  "https://example.com/sample_data/industrial_sensors_6months.csv"
```

## Complete Analysis Workflow

### Step 1: Data Import and Quality Assessment

```bash
# Import multi-sensor data
chronos import --file industrial_sensors.csv \
  --time-column timestamp \
  --value-columns "temp_inlet,temp_outlet,temp_bearing,pressure_suction,pressure_discharge,vibration_motor,vibration_compressor,flow_rate,power_consumption" \
  --frequency "1min" \
  --validate \
  --output sensors_processed.csv

# Assess data quality for each sensor
chronos stats --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --missing-value-analysis \
  --outlier-detection \
  --output data_quality_report.json
```

**Data Quality Results**:
```json
{
  "total_observations": 259200,
  "sensors": {
    "temp_bearing": {
      "missing_percentage": 0.02,
      "outliers_detected": 847,
      "drift_detected": false,
      "sensor_health": "good"
    },
    "vibration_motor": {
      "missing_percentage": 0.15,
      "outliers_detected": 1205,
      "drift_detected": true,
      "sensor_health": "degraded"
    }
  }
}
```

### Step 2: Baseline Establishment

```bash
# Establish normal operating baselines (first 30 days)
chronos stats --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --time-range "2023-01-01:2023-01-31" \
  --include-distributions \
  --output baseline_stats.json

# Create control limits for each sensor
chronos stats --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --control-limits \
  --confidence-level 0.99 \
  --output control_limits.json
```

**Baseline Establishment**:
```json
{
  "baseline_period": "2023-01-01 to 2023-01-31",
  "normal_ranges": {
    "temp_bearing": {
      "mean": 38.5,
      "std": 2.1,
      "control_limits": [32.1, 44.9],
      "warning_limits": [35.2, 41.8]
    },
    "vibration_motor": {
      "mean": 0.18,
      "std": 0.05,
      "control_limits": [0.05, 0.31],
      "warning_limits": [0.11, 0.25]
    }
  }
}
```

### Step 3: Trend Analysis and Degradation Detection

```bash
# Analyze long-term trends for each sensor
chronos trend --file sensors_processed.csv \
  --time-column timestamp \
  --value-column temp_bearing \
  --method comprehensive \
  --change-point-detection \
  --output bearing_temp_trend.json

# Analyze vibration trends (key indicator of mechanical wear)
chronos trend --file sensors_processed.csv \
  --time-column timestamp \
  --value-column vibration_motor \
  --method comprehensive \
  --decomposition stl \
  --change-point-detection \
  --output vibration_trend.json

# Multi-sensor trend correlation
chronos correlate --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns "temp_bearing,vibration_motor,vibration_compressor" \
  --rolling-correlation \
  --window 1440 \
  --output degradation_correlation.json
```

**Trend Analysis Findings**:
```json
{
  "bearing_temperature": {
    "trend_direction": "upward",
    "trend_strength": 0.34,
    "change_points": ["2023-02-15", "2023-04-22"],
    "degradation_rate": "0.08°C/month"
  },
  "vibration_correlation": {
    "motor_compressor": 0.67,
    "increasing_correlation": true,
    "coupling_degradation": "moderate"
  }
}
```

### Step 4: Anomaly Detection and Fault Identification

```bash
# Real-time anomaly detection for critical sensors
chronos anomaly --file sensors_processed.csv \
  --time-column timestamp \
  --value-column temp_bearing \
  --method ensemble \
  --seasonal-adjustment \
  --sensitivity 0.01 \
  --streaming \
  --output bearing_anomalies.json

# Multi-variate anomaly detection
chronos anomaly --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns "temp_bearing,vibration_motor,pressure_discharge" \
  --method multivariate \
  --contextual \
  --output system_anomalies.json

# Pattern-based fault detection
chronos anomaly --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --method pattern-matching \
  --fault-signatures fault_patterns.json \
  --output fault_detection.json
```

**Anomaly Detection Results**:
```json
{
  "critical_anomalies": [
    {
      "timestamp": "2023-03-15 14:23:00",
      "type": "temperature_spike",
      "sensors": ["temp_bearing"],
      "severity": "high",
      "probable_cause": "lubrication_issue"
    },
    {
      "timestamp": "2023-04-22 09:45:00",
      "type": "vibration_pattern",
      "sensors": ["vibration_motor", "vibration_compressor"],
      "severity": "medium",
      "probable_cause": "bearing_wear"
    }
  ],
  "total_anomalies": 156,
  "false_positive_rate": 0.03
}
```

### Step 5: Predictive Maintenance Modeling

```bash
# Time-to-failure prediction using degradation models
chronos forecast --file sensors_processed.csv \
  --time-column timestamp \
  --value-column temp_bearing \
  --method degradation-model \
  --failure-threshold 50.0 \
  --confidence-intervals \
  --output ttf_bearing_temp.json

# Remaining useful life (RUL) estimation
chronos forecast --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns "vibration_motor,temp_bearing" \
  --method survival-analysis \
  --horizon 4320 \
  --output rul_estimation.json

# Maintenance window optimization
chronos forecast --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --method maintenance-optimization \
  --maintenance-costs maintenance_costs.json \
  --output optimal_schedule.json
```

**Predictive Maintenance Results**:
```json
{
  "remaining_useful_life": {
    "bearing_system": {
      "estimated_rul": "67 days",
      "confidence_interval": [52, 84],
      "failure_probability": {
        "30_days": 0.05,
        "60_days": 0.42,
        "90_days": 0.78
      }
    }
  },
  "optimal_maintenance": {
    "recommended_date": "2023-06-15",
    "cost_savings": 23500,
    "downtime_reduction": "65%"
  }
}
```

### Step 6: Performance Monitoring and KPIs

```bash
# Calculate equipment performance metrics
chronos stats --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns "flow_rate,power_consumption,pressure_discharge" \
  --performance-metrics \
  --efficiency-analysis \
  --output performance_kpis.json

# Energy efficiency tracking
chronos correlate --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns "flow_rate,power_consumption" \
  --efficiency-analysis \
  --baseline-period "2023-01-01:2023-01-31" \
  --output energy_efficiency.json

# Overall Equipment Effectiveness (OEE)
chronos stats --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --oee-calculation \
  --availability-threshold 95 \
  --quality-threshold 99 \
  --output oee_metrics.json
```

**Performance Metrics**:
```json
{
  "efficiency_metrics": {
    "energy_efficiency": {
      "baseline": 1.65,
      "current": 1.58,
      "degradation": "4.2%"
    },
    "oee": {
      "availability": 0.94,
      "performance": 0.87,
      "quality": 0.98,
      "overall": 0.80
    }
  },
  "trending": {
    "efficiency_trend": "declining",
    "maintenance_impact": "high"
  }
}
```

### Step 7: Condition Monitoring Dashboard

```bash
# Create real-time monitoring dashboard
chronos plot --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns "temp_bearing,vibration_motor" \
  --type condition-monitoring \
  --control-limits control_limits.json \
  --anomalies bearing_anomalies.json \
  --output condition_dashboard.html

# Generate sensor health heatmap
chronos plot --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --type sensor-health-heatmap \
  --output sensor_health.png

# Create degradation trend plots
chronos plot --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns "temp_bearing,vibration_motor" \
  --type degradation-trends \
  --failure-predictions ttf_bearing_temp.json \
  --output degradation_analysis.png
```

### Step 8: Alert System Configuration

```bash
# Configure automated alerts
chronos config set alerts.enable true
chronos config set alerts.email "maintenance@company.com"
chronos config set alerts.thresholds '{
  "temp_bearing": {"critical": 45, "warning": 42},
  "vibration_motor": {"critical": 0.3, "warning": 0.25}
}'

# Set up real-time monitoring
chronos anomaly --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --streaming \
  --alert-system \
  --output real_time_alerts.log
```

### Step 9: Maintenance Integration

```bash
# Generate maintenance work orders
chronos report --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --template maintenance \
  --include-forecasts \
  --include-recommendations \
  --output maintenance_report.pdf

# Cost-benefit analysis
chronos stats --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --cost-analysis \
  --maintenance-costs maintenance_costs.json \
  --downtime-costs downtime_costs.json \
  --output cost_benefit.json
```

**Maintenance Recommendations**:
```json
{
  "immediate_actions": [
    {
      "priority": "high",
      "component": "motor_bearing",
      "action": "lubrication_check",
      "estimated_cost": 500,
      "risk_reduction": "high"
    }
  ],
  "scheduled_maintenance": [
    {
      "date": "2023-06-15",
      "component": "bearing_assembly",
      "action": "replacement",
      "estimated_cost": 15000,
      "expected_rul_extension": "18_months"
    }
  ]
}
```

### Step 10: Historical Analysis and Learning

```bash
# Analyze historical failure patterns
chronos stats --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --failure-analysis \
  --maintenance-logs maintenance_history.json \
  --output failure_pattern_analysis.json

# Model validation and improvement
chronos forecast --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --model-validation \
  --holdout-period 30 \
  --output model_performance.json
```

## Key Insights and Actionable Recommendations

### Equipment Health Assessment

**Current Status**:
- Motor bearing showing early signs of wear (4.2% efficiency degradation)
- Vibration levels trending upward but within acceptable limits
- Temperature control functioning properly
- Overall equipment effectiveness: 80% (industry benchmark: 85%)

**Risk Assessment**:
- High probability (78%) of bearing failure within 90 days without intervention
- Estimated downtime without predictive maintenance: 72 hours
- Potential production loss: $235,000

### Maintenance Strategy Optimization

**Immediate Actions** (Next 7 days):
1. Lubrication system inspection and service
2. Vibration sensor recalibration
3. Thermal imaging inspection of bearing assembly

**Planned Maintenance** (Next 60 days):
1. Motor bearing replacement (June 15, 2023)
2. Coupling alignment verification
3. Cooling system optimization

**Long-term Improvements**:
1. Implement condition-based maintenance protocols
2. Upgrade to predictive analytics platform
3. Install additional vibration sensors for better coverage

### Cost-Benefit Analysis

**Current Maintenance Approach**:
- Annual maintenance cost: $125,000
- Unplanned downtime: 156 hours/year
- Total annual cost: $892,000

**Predictive Maintenance Approach**:
- Annual maintenance cost: $95,000
- Unplanned downtime: 24 hours/year
- Total annual cost: $245,000
- **Net savings: $647,000/year (72% reduction)**

## Automation and Integration

### Real-time Monitoring Script

```bash
#!/bin/bash
# continuous_monitoring.sh

while true; do
    # Fetch latest sensor data
    chronos import --file /sensors/latest_data.csv --append --streaming

    # Run anomaly detection
    chronos anomaly --file /sensors/latest_data.csv \
      --time-column timestamp \
      --value-columns all \
      --method ensemble \
      --streaming \
      --alert-threshold 2.0

    # Update predictions
    chronos forecast --file /sensors/latest_data.csv \
      --value-columns "temp_bearing,vibration_motor" \
      --method degradation-model \
      --horizon 1440 \
      --output /predictions/latest_forecast.json

    # Check for critical alerts
    if [ -f /alerts/critical.flag ]; then
        # Send immediate notification
        curl -X POST "https://api.company.com/alerts" \
          -H "Content-Type: application/json" \
          -d @/alerts/latest_alert.json
    fi

    sleep 60  # Check every minute
done
```

### Integration with CMMS

```python
# Python integration with Computerized Maintenance Management System
import requests
import json
import subprocess

def update_cmms_with_predictions():
    """Update CMMS with latest failure predictions"""

    # Get latest RUL predictions
    result = subprocess.run([
        'chronos', 'forecast',
        '--file', 'sensors_data.csv',
        '--method', 'degradation-model',
        '--output', 'latest_rul.json'
    ], capture_output=True)

    with open('latest_rul.json', 'r') as f:
        predictions = json.load(f)

    # Update CMMS work orders
    for equipment, rul_data in predictions.items():
        if rul_data['days_remaining'] < 60:
            work_order = {
                'equipment_id': equipment,
                'priority': 'high' if rul_data['days_remaining'] < 30 else 'medium',
                'estimated_failure_date': rul_data['failure_date'],
                'recommended_action': rul_data['maintenance_action'],
                'cost_estimate': rul_data['maintenance_cost']
            }

            requests.post('https://cmms.company.com/api/work_orders',
                         json=work_order)

# Schedule to run daily
update_cmms_with_predictions()
```

## Industry 4.0 Integration

### Digital Twin Implementation

```bash
# Create digital twin model
chronos forecast --file sensors_processed.csv \
  --time-column timestamp \
  --value-columns all \
  --method digital-twin \
  --physics-model compressor_model.json \
  --output digital_twin_state.json

# Simulate "what-if" scenarios
chronos forecast --file sensors_processed.csv \
  --method simulation \
  --scenarios "load_increase_20%,temperature_rise_5C" \
  --output scenario_analysis.json
```

### Machine Learning Model Deployment

```python
# Deploy ML model for edge computing
from chronos import StreamingAnomalyDetector

detector = StreamingAnomalyDetector(
    model_path='trained_model.pkl',
    sensors=['temp_bearing', 'vibration_motor'],
    threshold=0.05
)

# Real-time inference
for sensor_reading in sensor_stream:
    anomaly_score = detector.predict(sensor_reading)
    if anomaly_score > threshold:
        send_alert(sensor_reading, anomaly_score)
```

## ROI and Business Impact

### Quantified Benefits

**Direct Cost Savings**:
- Maintenance cost reduction: $30,000/year
- Unplanned downtime elimination: $580,000/year
- Energy efficiency improvement: $37,000/year
- **Total Direct Savings: $647,000/year**

**Indirect Benefits**:
- Improved product quality and consistency
- Enhanced workplace safety
- Regulatory compliance documentation
- Knowledge capture and transfer
- **Estimated Indirect Value: $200,000/year**

### Implementation Timeline

**Phase 1** (Months 1-2): Sensor installation and data collection
**Phase 2** (Months 3-4): Baseline establishment and model development
**Phase 3** (Months 5-6): Predictive analytics implementation
**Phase 4** (Months 7+): Full deployment and optimization

**Payback Period**: 8 months
**5-Year NPV**: $2.8 million (assuming 8% discount rate)

This comprehensive industrial sensor monitoring example demonstrates how time series analysis can transform maintenance operations, reduce costs, and improve equipment reliability through data-driven insights.