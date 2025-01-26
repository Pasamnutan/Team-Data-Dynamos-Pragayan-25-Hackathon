// Flutter Application for Worker Safety Monitoring

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:io';

void main() {
  runApp(SafetyApp());
}

class SafetyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Worker Safety Monitoring',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: SafetyDashboard(),
    );
  }
}

class SafetyDashboard extends StatefulWidget {
  @override
  _SafetyDashboardState createState() => _SafetyDashboardState();
}

class _SafetyDashboardState extends State<SafetyDashboard> {
  double heartRate = 0;
  double locationRisk = 0;
  bool equipmentStatus = true;
  String riskScore = "";
  List<dynamic> ppeViolations = [];

  Future<void> fetchRiskPrediction() async {
    try {
      final response = await http.post(
        Uri.parse('http://127.0.0.1:5000/predict'), // Change to your server IP if not running locally
        body: {
          'heart_rate': heartRate.toString(),
          'location_risk': locationRisk.toString(),
          'equipment_status': equipmentStatus ? '1' : '0',
        },
      );
      final data = json.decode(response.body);
      setState(() {
        riskScore = data['risk_score'].toString();
        ppeViolations = data['ppe_violations'];
      });
    } catch (error) {
      print('Error fetching data: $error');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Worker Safety Dashboard')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text('Worker Safety Monitoring',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
            SizedBox(height: 20),
            TextField(
              decoration: InputDecoration(
                labelText: 'Heart Rate',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.number,
              onChanged: (value) {
                heartRate = double.tryParse(value) ?? 0;
              },
            ),
            SizedBox(height: 10),
            TextField(
              decoration: InputDecoration(
                labelText: 'Location Risk (0 to 1)',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.number,
              onChanged: (value) {
                locationRisk = double.tryParse(value) ?? 0;
              },
            ),
            SizedBox(height: 10),
            Row(
              children: [
                Text('Equipment Status: '),
                Switch(
                  value: equipmentStatus,
                  onChanged: (value) {
                    setState(() {
                      equipmentStatus = value;
                    });
                  },
                ),
              ],
            ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: fetchRiskPrediction,
              child: Text('Get Risk Prediction'),
            ),
            SizedBox(height: 20),
            Text('Risk Score: $riskScore',
                style: TextStyle(fontSize: 18, color: Colors.red)),
            Expanded(
              child: ListView.builder(
                itemCount: ppeViolations.length,
                itemBuilder: (context, index) {
                  return ListTile(
                    leading: Icon(Icons.warning, color: Colors.red),
                    title: Text('${ppeViolations[index]['name']} Violation'),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// Flask server code
// Save the below code as app.py and run it using Python

from flask import Flask, request, jsonify
import joblib
import torch
from PIL import Image
import io

app = Flask(__name__)

# Load hazard model
hazard_model = joblib.load('hazard_model.pkl')

# Load YOLOv5 model
model_ppe = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        heart_rate = float(request.form.get('heart_rate', 0))
        location_risk = float(request.form.get('location_risk', 0))
        equipment_status = int(request.form.get('equipment_status', 1))

        # Predict hazard risk
        risk_score = hazard_model.predict_proba([[heart_rate, location_risk, equipment_status]])[0][1]

        # Handle file input for PPE detection
        file = request.files['file'] if 'file' in request.files else None
        ppe_violations = []
        if file:
            img = Image.open(io.BytesIO(file.read()))
            results = model_ppe(img)
            ppe_violations = results.pandas().xyxy[0].to_dict(orient='records')

        return jsonify({
            'risk_score': risk_score,
            'ppe_violations': ppe_violations
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

