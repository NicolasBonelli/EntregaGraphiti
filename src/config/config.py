"""
Módulo para conectar y gestionar Graphiti con Neo4j Aura y Google Gemini
"""
from typing import Optional
import os
from dotenv import load_dotenv
from graphiti_core import Graphiti
from neo4j import GraphDatabase
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

load_dotenv()

class GraphitiConnector:
    """Clase para manejar la conexión y operaciones con Graphiti"""

    def __init__(self):
        """Inicializa la conexión con Graphiti, Neo4j Aura y Google Gemini usando variables de entorno"""
        # Google Gemini
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        self.gemini_model = os.getenv("GEMINI_MODEL")  
        self.gemini_embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL") 

        # Neo4j Aura
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = os.getenv("NEO4J_DATABASE")

        # Instancias
        self.gemini_client = self._setup_gemini()
        self.neo4j_driver = self._setup_neo4j()
        self._setup_neo4j_indexes()  # Crea índices al inicializar
        self.graphiti = self._setup_graphiti()

    def _setup_gemini(self):
        """Configura el cliente de Google Gemini"""
        try:
            return GeminiClient(
                config=LLMConfig(
                    api_key=self.gemini_api_key,
                    model=self.gemini_model
                )
            )
        except Exception as e:
            raise Exception(f"Error configurando Gemini: {e}")

    def _setup_neo4j(self) -> GraphDatabase.driver:
        """Configura la conexión con Neo4j Aura"""
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
            raise Exception(f"Error configurando Neo4j: {e}")

    def _setup_neo4j_indexes(self):
        """Crea los índices fulltext y regulares necesarios para Graphiti en Neo4j Aura"""
        try:
            with self.neo4j_driver.session(database=self.neo4j_database) as session:
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

                print("Todos los índices necesarios están configurados.")
        except Exception as e:
            raise Exception(f"Error creando índices en Neo4j: {e}")

    def _setup_graphiti(self) -> Graphiti:
        """Configura Graphiti con Neo4j Aura y Google Gemini"""
        try:
            llm_config = LLMConfig(
                api_key=self.gemini_api_key,
                model=self.gemini_model, 
                small_model=self.gemini_model  
            )
            graphiti = Graphiti(
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                llm_client=GeminiClient(
                    config=llm_config
                ),
                embedder=GeminiEmbedder(
                    config=GeminiEmbedderConfig(
                        api_key=self.gemini_api_key,
                        embedding_model=self.gemini_embedding_model
                    )
                ),
                cross_encoder=GeminiRerankerClient(
                    config=LLMConfig(
                        api_key=self.gemini_api_key,
                        model=self.gemini_model
                    )
                )
            )
            return graphiti
        except Exception as e:
            raise Exception(f"Error configurando Graphiti: {e}")

    


# --- MAIN DE PRUEBA ---
if __name__ == "__main__":
    print("Probando clase GraphitiConnector...")
    connector = GraphitiConnector()

    # Test Neo4j
    print("Probando conexión a Neo4j Aura...")
    try:
        with connector.neo4j_driver.session(database=connector.neo4j_database) as session:
            result = session.run("RETURN 1 AS test")
            value = result.single()["test"]
            print(f"Conexión a Neo4j Aura exitosa. Resultado: {value}")
    except Exception as e:
        print(f"Error en la conexión a Neo4j Aura: {e}")

    # Test Gemini
    print("Probando conexión a Google Gemini...")
    try:
        prompt = "Dame una lista de 3 colores primarios."
        response = connector.gemini_client.generate(prompt=prompt)
        print("Conexión a Gemini exitosa. Respuesta:")
        print(response)
    except Exception as e:
        print(f"Error en la conexión a Gemini: {e}")