# KrishiSahyog

ML-powered agriculture advisory web application for Indian farmers. React + FastAPI + JWT auth + optional ML models.

## Ports

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

## Troubleshooting

### ECONNREFUSED / Proxy errors
- Ensure backend is running: `curl http://127.0.0.1:8000/health`
- Backend must bind to `127.0.0.1:8000` (not just `localhost` on some setups)
- Frontend proxy targets `http://127.0.0.1:8000`

### Backend exits silently
- Check Python version: `python --version` (3.10 or 3.11 recommended)
- Install deps: `cd backend && pip install -r requirements.txt`
- Run directly: `cd backend && uvicorn main:app --host 127.0.0.1 --port 8000`

### Database errors
- SQLite creates `backend/krishisahyog.db` on first run
- Ensure `backend/` is writable

### Chatbot returns 503
- Set `OPENAI_API_KEY` or `GEMINI_API_KEY` in `backend/.env`
- Restart backend after changing .env

### Docker build fails
- Ensure `backend/.env` exists (copy from `.env.example`)
- For frontend: `npm run build` must succeed locally first

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
