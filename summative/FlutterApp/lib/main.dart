// Maternal Health Risk Predictor
// -------------------------------
// Single-page Flutter app that collects the six vitals required by the
// Maternal Risk Prediction API, calls POST /predict, and displays the
// resulting risk score / band, or a clear error message.

import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MaternalRiskApp());
}

class MaternalRiskApp extends StatelessWidget {
  const MaternalRiskApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Maternal Health Risk Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorSchemeSeed: const Color(0xFF7A4EAB),
        useMaterial3: true,
        scaffoldBackgroundColor: const Color(0xFFF7F4FB),
      ),
      home: const PredictorPage(),
    );
  }
}

class PredictorPage extends StatefulWidget {
  const PredictorPage({super.key});

  @override
  State<PredictorPage> createState() => _PredictorPageState();
}

class _PredictorPageState extends State<PredictorPage> {
  // ---------------------------------------------------------------------
  // >>> CONFIGURE THE API URL HERE <<<
  // Replace with your deployed Render URL, e.g.
  //   https://maternal-risk-api.onrender.com/predict
  // While developing against a local FastAPI server:
  //   Android emulator -> http://10.0.2.2:8000/predict
  //   iOS simulator / web -> http://127.0.0.1:8000/predict
  //   physical device -> http://<your-computer-LAN-ip>:8000/predict
  // ---------------------------------------------------------------------
  static const String apiBaseUrl = "https://maternal-risk-api-h32c.onrender.com";
  static const String predictPath = "/predict";

  final _formKey = GlobalKey<FormState>();

  final TextEditingController _ageController = TextEditingController();
  final TextEditingController _systolicController = TextEditingController();
  final TextEditingController _diastolicController = TextEditingController();
  final TextEditingController _bsController = TextEditingController();
  final TextEditingController _bodyTempController = TextEditingController();
  final TextEditingController _heartRateController = TextEditingController();

  bool _isLoading = false;
  String? _resultText;
  String? _riskBand;
  String? _errorText;

  @override
  void dispose() {
    _ageController.dispose();
    _systolicController.dispose();
    _diastolicController.dispose();
    _bsController.dispose();
    _bodyTempController.dispose();
    _heartRateController.dispose();
    super.dispose();
  }

  // Client-side range validation mirrors the API's Pydantic constraints, so
  // the user gets instant feedback before a network call is even made.
  String? _validateRange(String? value, String label, double min, double max) {
    if (value == null || value.trim().isEmpty) {
      return "$label is required";
    }
    final parsed = double.tryParse(value);
    if (parsed == null) {
      return "$label must be a number";
    }
    if (parsed < min || parsed > max) {
      return "$label must be between $min and $max";
    }
    return null;
  }

  Future<void> _predict() async {
    setState(() {
      _resultText = null;
      _riskBand = null;
      _errorText = null;
    });

    if (!_formKey.currentState!.validate()) {
      return; // inline field errors already shown by the Form
    }

    setState(() => _isLoading = true);

    final payload = {
      "age": int.parse(_ageController.text.trim()),
      "systolic_bp": double.parse(_systolicController.text.trim()),
      "diastolic_bp": double.parse(_diastolicController.text.trim()),
      "bs": double.parse(_bsController.text.trim()),
      "body_temp": double.parse(_bodyTempController.text.trim()),
      "heart_rate": double.parse(_heartRateController.text.trim()),
    };

    final uri = Uri.parse("$apiBaseUrl$predictPath");

    try {
      final response = await http
          .post(
            uri,
            headers: {"Content-Type": "application/json"},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 60));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          _resultText = (data["risk_score"] as num).toStringAsFixed(2);
          _riskBand = data["risk_band"] as String?;
        });
      } else if (response.statusCode == 422) {
        // Pydantic validation error from the API (e.g. an edge case the
        // client-side check above didn't catch).
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        setState(() {
          _errorText = "Invalid input: ${data["detail"] ?? response.body}";
        });
      } else {
        setState(() {
          _errorText = "Server error (${response.statusCode}). Please try again.";
        });
      }
    } on Exception catch (e) {
      setState(() {
        _errorText = "Could not reach the prediction server. "
            "Check your internet connection and the API URL.\n($e)";
      });
    } finally {
      setState(() => _isLoading = false);
    }
  }

  Widget _buildField({
    required TextEditingController controller,
    required String label,
    required String hint,
    required double min,
    required double max,
    bool isInteger = false,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: TextFormField(
        controller: controller,
        keyboardType: TextInputType.numberWithOptions(decimal: !isInteger),
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
          filled: true,
          fillColor: Colors.white,
        ),
        validator: (value) => _validateRange(value, label, min, max),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Maternal Health Risk Predictor"),
        centerTitle: true,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Text(
                  "Enter the patient's vitals below to estimate a "
                  "0-100 Maternal Risk Score.",
                  style: TextStyle(fontSize: 14, color: Colors.black54),
                ),
                const SizedBox(height: 20),
                _buildField(
                  controller: _ageController,
                  label: "Age (years)",
                  hint: "10 - 70",
                  min: 10,
                  max: 70,
                  isInteger: true,
                ),
                _buildField(
                  controller: _systolicController,
                  label: "Systolic BP (mmHg)",
                  hint: "70 - 200",
                  min: 70,
                  max: 200,
                ),
                _buildField(
                  controller: _diastolicController,
                  label: "Diastolic BP (mmHg)",
                  hint: "40 - 140",
                  min: 40,
                  max: 140,
                ),
                _buildField(
                  controller: _bsController,
                  label: "Blood Sugar (mmol/L)",
                  hint: "3.0 - 25.0",
                  min: 3.0,
                  max: 25.0,
                ),
                _buildField(
                  controller: _bodyTempController,
                  label: "Body Temperature (°F)",
                  hint: "95.0 - 106.0",
                  min: 95.0,
                  max: 106.0,
                ),
                _buildField(
                  controller: _heartRateController,
                  label: "Heart Rate (bpm)",
                  hint: "30 - 180",
                  min: 30,
                  max: 180,
                ),
                const SizedBox(height: 10),
                ElevatedButton(
                  onPressed: _isLoading ? null : _predict,
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  child: _isLoading
                      ? const SizedBox(
                          height: 20,
                          width: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2.5,
                            color: Colors.white,
                          ),
                        )
                      : const Text("Predict", style: TextStyle(fontSize: 16)),
                ),
                const SizedBox(height: 24),
                _buildResultArea(),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildResultArea() {
    if (_errorText != null) {
      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.red.shade50,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.red.shade200),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.error_outline, color: Colors.red),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                _errorText!,
                style: const TextStyle(color: Colors.red),
              ),
            ),
          ],
        ),
      );
    }

    if (_resultText != null) {
      final Color bandColor = _riskBand == "high risk"
          ? Colors.red
          : _riskBand == "mid risk"
              ? Colors.orange
              : Colors.green;

      return Container(
        width: double.infinity,
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: bandColor.withOpacity(0.08),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: bandColor.withOpacity(0.4)),
        ),
        child: Column(
          children: [
            const Text(
              "Predicted Maternal Risk Score",
              style: TextStyle(fontSize: 14, color: Colors.black54),
            ),
            const SizedBox(height: 6),
            Text(
              _resultText!,
              style: TextStyle(
                fontSize: 40,
                fontWeight: FontWeight.bold,
                color: bandColor,
              ),
            ),
            const SizedBox(height: 6),
            Text(
              (_riskBand ?? "").toUpperCase(),
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: bandColor,
                letterSpacing: 1.1,
              ),
            ),
          ],
        ),
      );
    }

    // Idle state placeholder.
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey.shade100,
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Text(
        "Prediction results will appear here.",
        textAlign: TextAlign.center,
        style: TextStyle(color: Colors.black45),
      ),
    );
  }
}
