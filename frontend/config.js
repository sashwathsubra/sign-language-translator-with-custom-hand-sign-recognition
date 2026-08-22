let defaultBackendUrl = 'http://localhost:8000';
if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
    defaultBackendUrl = 'https://sign-language-translator-with-custom.onrender.com';
}
window.__BACKEND_URL__ = window.__BACKEND_URL__ || defaultBackendUrl;
