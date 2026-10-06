from portablellm.services.config_service import ConfigService
from portablellm.services.llm_service import LLMService
from portablellm.services.embedding_service import EmbeddingService

config = ConfigService()
llm_service = LLMService(config)
embedding_service = EmbeddingService(config)
