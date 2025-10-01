
"""
Módulo para conectar y gestionar Graphiti con Neo4j y Azure OpenAI
"""
from openai import AsyncAzureOpenAI
from typing import List, Optional
import os
from dotenv import load_dotenv
from graphiti_core import Graphiti
from neo4j import GraphDatabase
from graphiti_core.llm_client import LLMConfig, OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI

load_dotenv(override=True)

class GraphitiConnector:
    """Clase para manejar la conexión y operaciones con Graphiti"""

    def __init__(self):
        """Inicializa la conexión con Graphiti, Neo4j y Azure OpenAI usando variables de entorno"""
        # Azure OpenAI
        self.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        # Azure OpenAI para chat
        self.azure_chat_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        self.azure_chat_api_version = os.getenv("AZURE_OPENAI_API_CHAT_VERSION")
        
        self.embedding_api_version = os.getenv("AZURE_OPENAI_EMBEDDING_API_VERSION")
        self.embedding_endpoint = os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT")
        self.azure_embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

        # Neo4j
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = os.getenv("NEO4J_DATABASE")

        # Instancias
        self.azure_client,self.azure_embedding_client,self.azure_graphity_client,self.azure_chat = self._setup_azure_openai()
        self.neo4j_driver = self._setup_neo4j()
        self._setup_neo4j_indexes()  # Crea índices al inicializar
        self.graphiti = self._setup_graphiti()

    def _setup_neo4j_indexes(self):
        """Crea los índices fulltext y regulares necesarios para Graphiti en Neo4j Aura"""
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
                # Tus índices existentes...
                # Índice fulltext para nodos (name y summary)
                session.run("""
                    CREATE FULLTEXT INDEX node_name_and_summary IF NOT EXISTS
                    FOR (n:Node) ON EACH [n.name, n.summary]
                """)
                print("Índice fulltext 'node_name_and_summary' creado o ya existe.")

                # Índice fulltext para entidades (name y description)
                session.run("""
                    CREATE FULLTEXT INDEX entity_search IF NOT EXISTS
                    FOR (n:Entity) ON EACH [n.name, n.description]
                """)
                print("Índice fulltext 'entity_search' creado o ya existe.")

                # Índice regular para UUIDs
                session.run("""
                    CREATE INDEX node_uuid IF NOT EXISTS
                    FOR (n:Node) ON (n.uuid)
                """)
                print("Índice regular 'node_uuid' creado o ya existe.")

                # Nuevo índice fulltext para edges (name y fact)
                session.run("""
                    CREATE FULLTEXT INDEX edge_name_and_fact IF NOT EXISTS
                    FOR ()-[r:RELATES_TO]-() ON EACH [r.name, r.fact]
                """)
                print("Índice fulltext 'edge_name_and_fact' creado o ya existe.")

                print("Todos los índices necesarios están configurados.")
        except Exception as e:
            raise Exception(f"Error creando índices en Neo4j: {e}")

    def _setup_azure_openai(self):
        """Configura el cliente de Azure OpenAI"""
        try:
            azure_client = AzureOpenAI(
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
                azure_endpoint=self.azure_endpoint
            )
            azure_graphiti_client = AsyncAzureOpenAI(
                api_key=self.azure_api_key,
                api_version=self.azure_api_version,
                azure_endpoint=self.azure_endpoint
            )
            embedding_client_azure = AsyncAzureOpenAI(
                api_key=self.azure_api_key,
                api_version=self.embedding_api_version,
                azure_endpoint=self.embedding_endpoint
            )
            azure_chat_client = AzureChatOpenAI(
                azure_endpoint=self.azure_endpoint,
                api_key=self.azure_api_key,
                deployment_name=self.azure_chat_deployment_name,
                api_version=self.azure_chat_api_version
            )
            return azure_client,embedding_client_azure,azure_graphiti_client,azure_chat_client
        except Exception as e:
            raise

    def _setup_neo4j(self) -> GraphDatabase.driver:
        """Configura la conexión con Neo4j"""
        try:
            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Verificar conexión
            with driver.session(database=self.neo4j_database) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            return driver
        except Exception as e:
            raise

    def _setup_graphiti(self) -> Graphiti:
        """Configura Graphiti con Neo4j y Azure OpenAI"""
        try:
            llm_config = LLMConfig(
                small_model=self.azure_deployment_name,  
                model=self.azure_deployment_name  
            )
            graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=OpenAIClient(
                    config=llm_config,
                    client=self.azure_graphity_client
                ),
                embedder=OpenAIEmbedder(
                    config=OpenAIEmbedderConfig(
                        embedding_model=self.azure_embedding_deployment  
                    ),
                    client=self.azure_embedding_client
                ),
                cross_encoder=OpenAIRerankerClient(
                    config=LLMConfig(model=llm_config.small_model),  
                    client=self.azure_graphity_client
                )
            )
            return graphiti
        except Exception as e:
            raise

    def get_openai_client_chat(self):
        """
        Returns the OpenAI model configuration for external usage.
        """
        return self.azure_chat


# --- MAIN DE PRUEBA ---
if __name__ == "__main__":
    print("Probando clase GraphitiConnector...")
    connector = GraphitiConnector()

    # Test Neo4j
    print("Probando conexión a Neo4j...")
    try:
        with connector.neo4j_driver.session(database=connector.neo4j_database) as session:
            result = session.run("RETURN 1 AS test")
            value = result.single()["test"]
            print(f"Conexión a Neo4j exitosa. Resultado: {value}")
    except Exception as e:
        print(f"Error en la conexión a Neo4j: {e}")

    # Test Azure OpenAI
    print("Probando conexión a Azure OpenAI...")
    try:
        prompt = "Dame una lista de 3 colores primarios."
        response = connector.azure_client.chat.completions.create(
            model=connector.azure_deployment_name,
            messages=[{"role": "user", "content": prompt}]
        )
        print("Conexión a Azure OpenAI exitosa. Respuesta:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error en la conexión a Azure OpenAI: {e}")