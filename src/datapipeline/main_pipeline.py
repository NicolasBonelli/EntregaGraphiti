import sys
from pathlib import Path
import asyncio
from src.datapipeline.extract_text import extract_text_from_pdf
from src.datapipeline.add_episodes import add_episodes_to_graphiti


def main(pdf_url: str):
    """
    Orquesta el pipeline completo:
    1. Extrae el texto del PDF desde la URL usando extract_text_from_pdf.
    2. Llama a add_episodes_to_graphiti con la ruta del texto extraído.
    """
    # Genera el nombre de salida en base al PDF
    pdf_path = Path(pdf_url)
    output_name = pdf_path.stem + "_extracted.txt"
    output_text_path = Path("data/output") / output_name

    # Paso 1: Extraer texto del PDF
    print(f"Extrayendo texto desde: {pdf_url}")
    extract_text_from_pdf(pdf_path, output_text_path)

    # Paso 2: Agregar episodios a Graphiti
    print(f"Agregando episodios desde: {output_text_path}")
    asyncio.run(add_episodes_to_graphiti(output_text_path))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python main_pipeline.py <ruta_o_url_pdf>")
        sys.exit(1)
    pdf_url = sys.argv[1]
    main(pdf_url)
