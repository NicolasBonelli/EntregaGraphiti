import asyncio
import os
from dotenv import load_dotenv
from src.agent.agent import create_graphiti_agent

# Carga variables de entorno
load_dotenv()


async def run_chatbot():
    """
    Ejecuta el chatbot con GraphRAG usando el agente ReAct.
    """
    
    # Crea el agente
    print("Inicializando agente GraphRAG...\n")
    agent_executor = create_graphiti_agent()
    
    print("=" * 60)
    print("Chatbot GraphRAG con Graphiti + Neo4j")
    print("=" * 60)
    print("Escribe 'salir' para terminar\n")
    
    # Loop de conversación
    while True:
        try:
            # Obtiene pregunta del usuario
            user_question = input("Tu pregunta: ").strip()
            
            if not user_question:
                continue
            
            if user_question.lower() in ['salir', 'exit', 'quit']:
                print("¡Hasta luego!")
                break
            
            # Invoca al agente (async)
            print("\n🤖 Pensando...\n")
            response = await agent_executor.ainvoke({"input": user_question})
            
            # Muestra la respuesta
            print("\n" + "=" * 60)
            print("Respuesta:")
            print(response["output"])
            print("=" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n¡Hasta luego!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


def main():
    """Punto de entrada del chatbot"""
    asyncio.run(run_chatbot())


if __name__ == "__main__":
    main()