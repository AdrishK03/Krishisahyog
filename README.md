# KrishiSahyog 🌾

**ML-powered agriculture advisory web application empowering Indian farmers with data-driven insights.**

An intelligent agricultural assistant combining machine learning models, real-time data analytics, and expert advisory to help farmers make informed decisions about crop management, pest diagnosis, soil analysis, and market trends.

## 🎯 Features

- **🤖 AI-Powered Chatbot** - Interactive chatbot for agriculture queries and advice
- **🔍 Plant Disease Diagnosis** - ML-based plant disease detection using image analysis
- **🌱 Soil Analysis** - Soil health assessment and recommendations
- **📊 Market Insights** - Real-time agricultural market information
- **🌦️ Weather Updates** - Location-based weather forecasts
- **🔐 Secure Authentication** - JWT-based user authentication and authorization
- **📱 Responsive Design** - Mobile-friendly interface with Tailwind CSS
- **🌐 Multi-language Support** - Google Translate integration

## 📋 Tech Stack

### Backend
- **Framework:** FastAPI (Python 3.10+)
- **Database:** SQLite (configurable)
- **Authentication:** JWT (PyJWT)
- **ML Models:** TensorFlow/Keras, PyTorch, ONNX
- **API Documentation:** Auto-generated Swagger UI
- **Containerization:** Docker

### Frontend
- **Framework:** React 18+ with TypeScript
- **Build Tool:** Vite
- **Styling:** Tailwind CSS + shadcn/ui components
- **State Management:** React Context API
- **HTTP Client:** Axios
- **Package Manager:** npm

## 🌐 Service Ports

| Service   | Local Dev       | Production (Docker) |
|-----------|-----------------|---------------------|
| Frontend  | http://localhost:5173 | http://localhost (port 80) |
| Backend   | http://127.0.0.1:8000 | http://localhost:8000 (also proxied via /api) |

---

## Part A: Local Development (Single Command)

**Prerequisites:** Python 3.10+, Node.js 18+, npm

```bash
# Install backend dependencies once
cd backend && pip install -r requirements.txt && cd ..

# Copy env template (optional; defaults work for dev)
cp backend/.env.example backend/.env

# Run everything: backend + frontend + open browser
python run.py
```

- **Backend:** http://127.0.0.1:8000  
- **Frontend:** http://localhost:5173  
- **Health:** http://127.0.0.1:8000/health  
- Press **Ctrl+C** to stop both servers

---

## Part B: Manual Local Dev

**Backend:**
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Frontend (new terminal):**
```bash
npm install
npm run dev
```

Frontend runs on http://localhost:5173 and proxies `/api` to the backend.

---

## Part C: Production Deploy (Docker)

```bash
# Create env file (required)
cp backend/.env.example backend/.env
# Edit backend/.env: set JWT_SECRET_KEY, OPENAI_API_KEY or GEMINI_API_KEY

# Build and run
docker-compose up -d

# Frontend: http://localhost
# Backend API: http://localhost:8000
```

**Stopping:**
```bash
docker-compose down
```

---

## Environment Variables

### Backend (`backend/.env`)

| Variable        | Required | Description                                      |
|-----------------|----------|--------------------------------------------------|
| JWT_SECRET_KEY  | Yes (prod) | Secret for JWT signing. Use a long random string. |
| DATABASE_URL    | No       | Default: `sqlite:///./krishisahyog.db`           |
| CORS_ORIGINS    | No       | Comma-separated frontend URLs                    |
| OPENAI_API_KEY  | For chat | OpenAI API key for chatbot                       |
| GEMINI_API_KEY  | For chat | Alternative to OpenAI                            |

### Frontend

| Variable       | Required | Description                                      |
|----------------|----------|--------------------------------------------------|
| VITE_API_URL   | No       | Default: `/api` (proxied in dev, nginx in prod)  |

---

## 📁 Project Structure

```
Krishisahyog/
├── backend/                 # FastAPI application
│   ├── main.py             # Entry point
│   ├── auth/               # Authentication (JWT, password utils)
│   ├── chatbot/            # AI chatbot integration
│   ├── database/           # Database schema & models
│   ├── ml/                 # ML models & predictions
│   │   ├── plant_predictor.py
│   │   ├── soil_predictor.py
│   │   ├── disease_classes.py
│   │   └── treatments.py
│   ├── models/             # Pre-trained ML models (.pth, .keras, .onnx)
│   ├── Dockerfile          # Backend Docker image
│   └── requirements.txt     # Python dependencies
│
├── frontend/               # React/TypeScript application
│   ├── src/
│   │   ├── pages/         # Route pages (Login, Dashboard, Diagnosis, etc.)
│   │   ├── components/    # Reusable UI components
│   │   ├── contexts/      # React Context (Auth, etc.)
│   │   ├── hooks/         # Custom React hooks
│   │   ├── services/      # API service layer
│   │   └── lib/           # Utilities & Firebase config
│   ├── vite.config.ts     # Vite configuration with API proxy
│   ├── tailwind.config.ts # Tailwind CSS config
│   ├── Dockerfile.frontend# Frontend Docker image
│   └── package.json        # Node.js dependencies
│
├── docker-compose.yml      # Multi-container orchestration
└── README.md              # This file
```

---

## 🚀 Quick Start Guide

### Option 1: One-Command Setup (Recommended)

**Prerequisites:** Python 3.10+, Node.js 18+, npm

```bash
# Install backend dependencies once
cd backend && pip install -r requirements.txt && cd ..

# Copy environment config (optional; defaults work for dev)
cp backend/.env.example backend/.env

# Run everything with one command
python run.py
```

**Access:**
- **Frontend:** http://localhost:5173  
- **Backend:** http://127.0.0.1:8000  
- **API Docs:** http://127.0.0.1:8000/docs  
- **Health:** http://127.0.0.1:8000/health  

Press **Ctrl+C** to stop both servers.

---

### Option 2: Manual Local Development

**Backend:**
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Frontend (new terminal):**
```bash
npm install
npm run dev
```

Frontend runs on http://localhost:5173 and proxies `/api` to the backend.

---

### Option 3: Docker Deployment

```bash
# Create environment file
cp backend/.env.example backend/.env
# Edit backend/.env: set JWT_SECRET_KEY, OPENAI_API_KEY or GEMINI_API_KEY

# Build and run containers
docker-compose up -d

# Access
# Frontend: http://localhost
# Backend API: http://localhost:8000
```

**Stopping:**
```bash
docker-compose down
```

---

## ⚙️ Environment Configuration

### Backend (`backend/.env`)

| Variable        | Required | Default | Description |
|-----------------|----------|---------|-------------|
| `JWT_SECRET_KEY` | Yes (prod) | `'dev-secret'` | Secret for JWT signing. Use a long random string in production. |
| `DATABASE_URL` | No | `'sqlite:///./krishisahyog.db'` | Database connection URL. SQLite by default. |
| `CORS_ORIGINS` | No | `'*'` | Comma-separated frontend URLs allowed to access the API. |
| `OPENAI_API_KEY` | For chatbot | - | OpenAI API key for chatbot functionality. |
| `GEMINI_API_KEY` | For chatbot | - | Google Gemini API key (alternative to OpenAI). |

### Frontend Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VITE_API_URL` | No | `/api` | Backend API URL (proxied in dev, nginx in production). |

---

## 📡 API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - User login (returns JWT)
- `POST /auth/refresh` - Refresh JWT token
- `GET /auth/profile` - Get current user profile (protected)

### Chatbot
- `POST /chatbot/chat` - Send message to chatbot (protected)
- `GET /chatbot/history` - Get chat history (protected)

### Diagnosis
- `POST /diagnosis/predict` - Plant disease diagnosis from image
- `GET /diagnosis/results/{id}` - Get diagnosis results (protected)

### Health & Status
- `GET /health` - Health check (no auth required)
- `GET /docs` - Interactive API documentation (Swagger UI)

---

## 🔒 Security Features

- **JWT Authentication:** Secure token-based authentication for all protected endpoints
- **Password Security:** Bcrypt hashing for password storage
- **CORS Protection:** Configurable Cross-Origin Resource Sharing
- **Environment Secrets:** Sensitive keys managed via environment variables

---

## 🚨 Troubleshooting

### Connection Refused / Proxy Errors
```bash
# Check if backend is running
curl http://127.0.0.1:8000/health

# Ensure backend binds to 127.0.0.1:8000 (not localhost)
```

### Backend Exits Without Error
```bash
# Verify Python version (3.10+ required)
python --version

# Reinstall dependencies
cd backend && pip install -r requirements.txt --force-reinstall
```

### Database Errors
```bash
# SQLite file should be created automatically in backend/
# Ensure backend/ directory is writable
ls -la backend/krishisahyog.db
```

### Chatbot Returns 503 Error
- Verify `OPENAI_API_KEY` or `GEMINI_API_KEY` is set in `backend/.env`
- Restart backend after changing environment variables

### Docker Build Fails
- Ensure `backend/.env` exists: `cp backend/.env.example backend/.env`
- Test local build: `npm run build` in frontend/
- Check Docker daemon is running

---

## 🛠️ Development Tips

### Running Tests
```bash
cd backend
pytest tests/
```

### Building Frontend for Production
```bash
npm run build
# Output in dist/
```

### Viewing API Documentation
Visit http://127.0.0.1:8000/docs while backend is running.

### Debugging with VS Code
Set breakpoints in backend/main.py and use the debug configuration in .vscode/launch.json.

---

## 📦 Dependencies

### Backend
See [backend/requirements.txt](backend/requirements.txt) for complete list.
- fastapi
- sqlalchemy
- pyjwt
- tensorflow/pytorch
- python-dotenv

### Frontend  
See [frontend/package.json](frontend/package.json) for complete list.
- react
- shadcn/ui
- tailwindcss
- axios
- typescript

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature`
3. **Commit** changes with clear messages
4. **Push** to your fork: `git push origin feature/your-feature`
5. **Open** a Pull Request

### Code Standards
- Backend: Follow PEP 8, use type hints
- Frontend: Use ESLint configuration, TypeScript for type safety
- Both: Write meaningful commit messages

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 📞 Support

For issues, questions, or suggestions:
- Open an [Issue](../../issues) on GitHub
- Check [Troubleshooting](#troubleshooting) section above
- Review API documentation at `/docs` endpoint

---

## 🙏 Acknowledgments

Built with ❤️ for Indian farmers. Special thanks to:
- FastAPI & React communities
- ML model contributors
- All contributors and testers

---

## Project Structure

```
Krishisahyog/
├── run.py              # Single-command dev launcher
├── docker-compose.yml  # Production stack
├── Dockerfile.frontend # Frontend (nginx)
├── backend/
│   ├── main.py
│   ├── auth/
│   ├── database/
│   ├── ml/
│   ├── chatbot/
│   ├── models/         # Place .pkl ML models here
│   └── Dockerfile
└── src/                # React frontend
```

---

## License

MIT
