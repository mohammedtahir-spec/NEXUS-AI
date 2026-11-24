
# NEXUS AI - Frontend

The frontend of NEXUS AI, an enterprise document question-answering application powered by Retrieval-Augmented Generation (RAG).

## Technology

- React 19
- Vite
- Lucide React
- Nginx for Docker deployment

## Development

From the frontend directory:

```bash
npm install
npm run dev
```

The Vite development server runs at:

http://localhost:5173

## Production Build

Build the frontend:

```bash
npm run build
```

Run the production preview:

```bash
npm run preview
```

## Linting

```bash
npm run lint
```

## Docker

The frontend is containerized using Docker and served through Nginx.

When running the complete application with Docker Compose, access the frontend at:

http://localhost:8081

The frontend communicates with the FastAPI backend through the configured API proxy.
