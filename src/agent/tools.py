from datetime import datetime, timezone
from typing import Optional
from langchain.tools import tool
from src.agent.singleton_connection import get_connector
from graphiti_core.search.search_config_recipes import (
    NODE_HYBRID_SEARCH_RRF,
    EDGE_HYBRID_SEARCH_RRF
)

def get_graphiti():
    """Obtiene la instancia de Graphiti desde el singleton global"""
    return get_connector().graphiti


@tool
async def hybrid_search(query: str, limit: int = 10) -> str:
    """
    Búsqueda híbrida que combina similitud semántica (embeddings) y BM25 keyword search,
    rerankeada con Reciprocal Rank Fusion (RRF). Es la búsqueda más general y balanceada.
    
    Útil para: preguntas generales, búsqueda exploratoria, cuando no sabés qué tipo de 
    información necesitás (nodos, relaciones, o ambos).
    
    Args:
        query: La consulta de búsqueda (ej: "TechNova productos", "fundación empresa")
        limit: Número máximo de resultados (default: 10)
    
    Returns:
        Información combinada de nodos y relaciones relevantes
    """
    graphiti = get_graphiti()
    results = await graphiti.search(query=query, num_results=limit)
    
    if not results:
        return "No se encontró información relevante para la consulta."
    
    output = ["=== BÚSQUEDA HÍBRIDA (Semántica + Keyword) ==="]
    for item in results:
        if hasattr(item, 'name') and hasattr(item, 'summary'):
            # Es un nodo
            output.append(f"[ENTIDAD] {item.name}: {item.summary or 'Sin descripción'}")
        elif hasattr(item, 'fact'):
            # Es un edge/fact
            output.append(f"[RELACIÓN] {item.fact}")
    
    return "\n".join(output)



@tool
async def temporal_aware_search(
    query: str, 
    reference_time: Optional[str] = None,
    limit: int = 10
) -> str:
    """
    Búsqueda con consciencia temporal (point-in-time query). Permite consultar el estado
    del grafo en un momento específico del pasado, respetando valid_at/invalid_at de edges.
    
    Útil para: "¿Quién era el CEO en 2023?", "¿Qué productos existían al inicio?",
    queries históricas, evolución temporal de hechos.
    
    Args:
        query: La consulta de búsqueda
        reference_time: Fecha/hora ISO (ej: "2024-01-15T10:00:00Z"). Si es None, usa tiempo actual
        limit: Número máximo de resultados (default: 10)
    
    Returns:
        Información válida en el punto temporal especificado
    """
    graphiti = get_graphiti()
    
    # Parsear timestamp o usar actual
    if reference_time:
        try:
            ref_time = datetime.fromisoformat(reference_time.replace('Z', '+00:00'))
        except ValueError:
            return f"Error: timestamp inválido. Use formato ISO: YYYY-MM-DDTHH:MM:SSZ"
    else:
        ref_time = datetime.now(timezone.utc)
    
    # Búsqueda básica (sin reference_time, como fallback)
    results = await graphiti.search(
        query=query,
        num_results=limit * 2  # Traemos más para filtrar manualmente
    )
    
    if not results:
        return f"No se encontró información para '{query}' en {ref_time.strftime('%Y-%m-%d %H:%M:%S UTC')}."
    
    # Filtrado manual por valid_at/invalid_at
    filtered_results = []
    for item in results:
        if hasattr(item, 'fact'):  # Edge
            valid_at = getattr(item, 'valid_at', None)
            invalid_at = getattr(item, 'invalid_at', None)
            
            # Filtrar: válido si valid_at <= ref_time y (invalid_at > ref_time o NULL)
            if valid_at and invalid_at:
                if valid_at <= ref_time and (invalid_at is None or invalid_at > ref_time):
                    filtered_results.append(item)
            elif valid_at and valid_at <= ref_time:
                filtered_results.append(item)
            elif not valid_at:  # Si no tiene timestamps, incluir por default
                filtered_results.append(item)
        else:  # Nodo, incluir siempre (nodos no tienen temporalidad)
            filtered_results.append(item)
    
    filtered_results = filtered_results[:limit]  # Limitar después del filtro
    
    if not filtered_results:
        return f"No se encontró información válida para '{query}' en {ref_time.strftime('%Y-%m-%d %H:%M:%S UTC')}."
    
    output = [f"=== BÚSQUEDA TEMPORAL: {ref_time.strftime('%Y-%m-%d %H:%M:%S UTC')} ==="]
    for item in filtered_results:
        if hasattr(item, 'fact'):
            # Es un edge (EntityEdge)
            valid_info = ""
            if hasattr(item, 'valid_at'):
                valid_info = f" (válido desde: {item.valid_at})"
            output.append(f"[HECHO] {item.fact}{valid_info}")
        else:
            # Es un nodo (EntityNode)
            output.append(f"[NODO] {item.name}: {item.summary or 'Sin descripción'}")
    
    return "\n".join(output)


if __name__ == "__main__":
    import asyncio

    async def test_tools():
        print("🔎 Probando hybrid_search...")
        res1 = await hybrid_search.ainvoke({"query": "TechNova productos", "limit": 5})
        print(res1, "\n")

        print("🔎 Probando temporal_aware_search...")
        res2 = await temporal_aware_search.ainvoke(
            {"query": "CEO de TechNova", "reference_time": "2023-01-01T00:00:00Z", "limit": 5}
        )
        print(res2, "\n")

    asyncio.run(test_tools())
