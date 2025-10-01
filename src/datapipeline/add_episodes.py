import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import re

from graphiti_core.nodes import EpisodeType
from src.config.config_azure import GraphitiConnector

# Rutas
input_dir = Path("data/output/tech_nova_extracted.txt")
output_dir = Path("data/output_episodes/")

def chunk_text_by_sentence_pairs(text: str) -> List[str]:
    """
    Divide el texto en episodios, donde cada episodio contiene exactamente dos oraciones
    terminadas en punto. Usa regex para detectar oraciones.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip() and len(s) > 2]
    episodes = []
    for i in range(0, len(sentences) - 1, 2):
        episode = " ".join(sentences[i:i+2])
        if episode.strip():
            episodes.append(episode)
    if len(sentences) % 2 == 1:
        last_episode = sentences[-1]
        if last_episode.strip():
            episodes.append(last_episode)
    return episodes

async def add_episodes_to_graphiti(input_dir: Path):
    """
    Lee el archivo, lo divide en episodios (dos oraciones por episodio), y los agrega a Graphiti.
    """
    try:
        # Usa la instancia de Graphiti
        connector = GraphitiConnector()
        graphiti = connector.graphiti

        # Lee el contenido
        with open(input_dir, "r", encoding="utf-8") as file:
            full_text = file.read().strip()

        if not full_text:
            print("El archivo está vacío.")
            return

        # Divide en episodios
        episodes_text = chunk_text_by_sentence_pairs(full_text)
        if not episodes_text:
            print("No se encontraron episodios válidos.")
            return

        print(f"Se generaron {len(episodes_text)} episodios.")

        # Prepara metadatos
        source_description = "Extracto de PDF Tech Nova, dividido en pares de oraciones"
        reference_time = datetime.now(timezone.utc)

        input_name = input_dir.stem.replace('_extracted', '')
        for i, episode_text in enumerate(episodes_text, 1):
            episode_name = f"{input_name}_episode_{i}"
            await graphiti.add_episode(
                name=episode_name,
                episode_body=episode_text,
                source=EpisodeType.text,
                source_description=source_description,
                reference_time=reference_time
            )
            print(f"Agregado episodio: {episode_name} ({EpisodeType.text.value})")
            await asyncio.sleep(1)

        print(f"Se agregaron {len(episodes_text)} episodios al grafo.")

        # Guarda episodios como archivos
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, episode_text in enumerate(episodes_text, 1):
            chunk_file = output_dir / f"episode_{i}.txt"
            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(episode_text)
        print("Episodios guardados en data/output_episodes/episode_*.txt")

    except FileNotFoundError:
        print(f"Error: No se encontró {input_dir}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await graphiti.close()
        print("Conexión cerrada.")

if __name__ == "__main__":
    asyncio.run(add_episodes_to_graphiti(input_dir))