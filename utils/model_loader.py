import os
import sys
from dotenv import load_dotenv
from utils.config_loader import load_config
from logger.custom_logger import CustomLogger
from exception.custom_exception import RegulatoryRAGException
from langchain_groq import ChatGroq
from langchain_aws import BedrockEmbeddings
from utils.observability import langfuse_callback
#Initialize the logger
log= CustomLogger().get_logger(__name__)

class ModelLoader:

    def __init__(self):
        load_dotenv()
        self._validate_env()
        self.config=load_config()
        log.info("Configuration loaded successfully", config_keys=list(self.config.keys()))

    def _validate_env(self):
        """
        Validate necessary environment variables
        Ensure API keys exist.
        """
        required_vars=["GROQ_API_KEY", "AWS_SECRET_ACCESS_KEY","AWS_ACCESS_KEY_ID", "AWS_DEFAULT_REGION"]
        self.api_keys={key:os.getenv(key) for key in required_vars}
        missing=[k for k, v in self.api_keys.items() if not v]
        if missing:
            log.error("Missing environment variables", missing_vars=missing)
            raise RegulatoryRAGException("Missing ennvironment variables", sys)
        log.info("Environment variables validated", available_keys=[k for k in self.api_keys if self.api_keys[k]])

    def load_embeddings(self):
        """Load and return the embedding model"""
        try:
            log.info("Loading embedding model....")
            model_name= self.config["embedding_model"]["model_name"]
            region= os.getenv("AWS_DEFAULT_REGION")
            return BedrockEmbeddings(model_id=model_name, region_name=region)
        except Exception as e:
            log.error("Error loading embedding model", error=str(e))
            raise RegulatoryRAGException("Failed to load embedding model", sys)
    def load_llm(self):
        """Load and return the configured LLM Model"""
        llm_block= self.config["llm"]
        log.info("Loading LLM...")
        provider_key= os.getenv("LLM_PROVIDER", "groq") #Default will be considered as groq
        if provider_key not in llm_block:
            log.error("LLM provider not found in config", provider_key=provider_key)
            raise ValueError(f"Provider '{provider_key} not found in config'")
        llm_config= llm_block[provider_key]
        provider= llm_config.get("provider")
        model_name=llm_config.get("model_name")
        temperature= llm_config.get("temperature", 0.2)
        max_tokens= llm_config.get("max_output_tokens", 2048)

        log.info("Loading LLM", provider=provider, model=model_name, temperature=temperature, max_tokens=max_tokens)
        if provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_keys.get("GROQ_API_KEY"),
                temperature=temperature,
                callbacks=[langfuse_callback]
            )
        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")
        


# if __name__== "__main__":
#     loader= ModelLoader()

#     #Test embedding model loading
#     embeddings= loader.load_embeddings()
#     print(f"Embedding Model Loaded: {embeddings}")

#     #Test LLM Loading based on you YAML config
#     llm= loader.load_llm()
#     print(f"LLM Loaded: {llm}")

#     #Test the ModelLoader
#     result= llm.invoke("Hello, how are you?")
#     print(f"LLM Result: {result.content}")