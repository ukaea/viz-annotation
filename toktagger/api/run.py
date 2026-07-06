import uvicorn
import toktagger.api.config as config

if __name__ == "__main__":
    uvicorn.run(
        "toktagger.api.cli:create_app",
        factory=True,
        host=config.settings.server.host,
        port=config.settings.server.port,
        reload=config.settings.server.reload,
    )
