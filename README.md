🌾 KrishiSahyog - Smart Agricultural Advisory SystemAn AI-powered agricultural advisory platform that empowers farmers with real-time crop monitoring, multi-crop disease detection, IoT sensor integration, market insights, and personalized farming recommendations in multiple languages.
📋 Table of ContentsFeaturesTechnology StackSystem ArchitectureInstallationUsageMulti-Crop Disease DetectionIoT IntegrationAPI DocumentationScreenshotsContributingLicenseContact

✨Features
🤖 AI-Powered Disease DetectionMulti-Crop Support: Separate trained models for Rice, Wheat, Tomato, and Potato.95%+ Accuracy: Deep learning CNN models for precise disease identification.Instant Diagnosis: Upload plant images and get results within seconds.Treatment Recommendations: Specific pesticide and care instructions for each disease.Confidence Scoring: Know the reliability of each diagnosis.
📊 Real-Time MonitoringIoT Sensor Integration: Connect Raspberry Pi/Arduino sensors.Soil Health Metrics: pH, moisture, temperature, NPK levels.Live Data Updates: WebSocket-powered real-time dashboard.Historical Analysis: Track soil health trends over time.Automated Alerts: Get notified of critical conditions.
🌤️ Weather IntelligenceLocalized Forecasts: 5-day weather predictions.Agricultural Alerts: Rainfall, temperature, humidity warnings.Crop-Specific Advice: Weather-based farming recommendations.API Integration: OpenWeatherMap integration.💰 Market InsightsReal-Time Prices: Current crop market rates.Price Trends: Historical price analysis and predictions.Quality Grading: A, B, and C grade pricing.Multiple Markets: Track prices across different locations.
🗣️ Multi-Language Support3 Languages: English, Hindi (हिंदी), Bengali (বাংলা).Voice Assistant: Speech-to-text for hands-free operation.AI Chatbot: Multilingual farming advice assistant.Complete Translation: Every UI element translated.

🏗️ System Architecture

┌─────────────────────────────────────────────────────────────┐
│                 USER INTERFACE LAYER (KrishiSahyog)         │
│  Web Browser │ Mobile App │ Voice Interface │ Chat Bot     │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   APPLICATION LAYER                         │
│  Flask Server │ SocketIO │ RESTful APIs │ WebSocket        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   PROCESSING LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ AI Disease   │  │ IoT Data     │  │ Weather &    │      │
│  │ Detection    │  │ Processing   │  │ Market API   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                      DATA LAYER                             │
│  SQLite/PostgreSQL │ File Storage │ Cache │ Logs           │
└─────────────────────────────────────────────────────────────┘
🚀 InstallationPrerequisitesPython 3.8 or higherGitVirtual environment toolStep 1: Clone RepositoryBashgit clone https://github.com/yourusername/krishisahyog.git
cd krishisahyog
Step 2: Set Up EnvironmentBash# Create and activate virtual environment
python -m venv venv
source venv/bin/activate # Linux/Mac
# or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
Step 3: Run ApplicationBashpython app.py
Access the application at: http://localhost:5000🌾 Multi-Crop Disease DetectionCropDiseases DetectedModel Accuracy🌾 RiceBrown Spot, Leaf Blight, Neck Blast, Healthy95%+🌾 WheatBrown Rust, Yellow Rust, Leaf Blight, Healthy93%+🍅 TomatoEarly Blight, Late Blight, Bacterial Spot, Leaf Curl, Healthy96%+🥔 PotatoEarly Blight, Late Blight, Healthy94%+🔌 IoT IntegrationWiring Diagram Concept:Hardware Requirements:Raspberry Pi 4B or Arduino Uno/NodeMCUADS1115 16-bit ADC ModulepH, Soil Moisture, and Temperature (DS18B20) Sensors📸 ScreenshotsHomepageDashboard🤝 ContributingWe welcome contributions to KrishiSahyog!Fork the repository.Create your feature branch (git checkout -b feature/NewFeature).Commit your changes (git commit -m 'Add NewFeature').Push to the branch (git push origin feature/NewFeature).Open a Pull Request.
