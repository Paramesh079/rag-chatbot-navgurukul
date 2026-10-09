"""Entry point — run the RAG Chatbot server."""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.api:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info",
    )
