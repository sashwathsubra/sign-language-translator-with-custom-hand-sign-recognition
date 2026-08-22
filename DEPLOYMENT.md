# Deployment guide

## Recommended free-tier plan

Use a small Python host for the WebSocket backend and a static host for the frontend.

- Backend: Render free web service
- Frontend: GitHub Pages or Cloudflare Pages

This combination is practical for a portfolio demo because:

- Render supports persistent WebSocket connections on its web service plan.
- GitHub Pages and Cloudflare Pages are free for static assets.
- The frontend is configured to use the same host for both local development and production via the current host name.

## Local development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install fastapi uvicorn websockets
   ```
2. Start the backend:
   ```bash
   python server.py
   ```
3. Serve the frontend files from the frontend directory with any static server, for example:
   ```bash
   python -m http.server 8080 --directory frontend
   ```
4. Open http://localhost:8080.

## Environment configuration

The frontend uses the current origin by default. For a production deployment, set a global variable before loading the app:

```html
<script>
  window.__BACKEND_URL__ = 'https://your-backend-url.example';
</script>
<script src="app.js"></script>
```

The backend reads the following environment variables:

- `ALLOWED_ORIGINS` for CORS/origin restriction
- `MAX_BUFFER_SIZE` to cap per-session backlog
- `IDLE_TIMEOUT_SECONDS` for session cleanup
- `MAX_CONCURRENT_SESSIONS` to cap active sessions on free-tier hosts
- `HOST` and `PORT` for binding

## Cold start and free-tier behavior

On a sleep-prone host, the frontend shows a clear "waking up the model, this can take up to a minute" state while the backend is waking up. Once connected, captions resume normally. If the backend is unavailable, the UI surfaces an error message instead of silently failing.

## Production deployment

### Backend (Render)

1. Create a new Web Service on Render.
2. Point it to this repository.
3. Set the start command to:
   ```bash
   python server.py
   ```
4. Render will expose a public URL such as https://your-service.onrender.com.
5. Update the frontend configuration if needed so the WebSocket URL points to the Render host.

### Frontend (GitHub Pages)

1. Publish the contents of the frontend directory as the site root.
2. If the backend is hosted elsewhere, make sure the frontend uses the deployed backend URL instead of localhost.

## Notes

- If the backend is cold-starting, the UI shows a clear message rather than failing silently.
- Keep the original Apache-2.0 license file intact when deploying.
