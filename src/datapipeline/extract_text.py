from docling.document_converter import DocumentConverter
from pathlib import Path

# Ruta al archivo PDF
pdf_path = Path("data/pdfs/tech_nova.pdf")

# Ruta al archivo de salida
output_text_path = Path("data/output/tech_nova_extracted.txt")

def extract_text_from_pdf(pdf_path: Path, output_text_path: Path):
    """
    Extrae texto de un PDF usando Docling y lo guarda en un archivo de texto (en formato Markdown).
    """
    try:
        # Inicializa el convertidor de Docling
        converter = DocumentConverter()

        # Convierte el PDF (puede ser una ruta local o URL)
        result = converter.convert(str(pdf_path))
        
        # Obtiene el documento procesado
        document = result.document
        
        # Exporta a Markdown (preserva estructura; usa export_to_dict() para JSON o ajusta para texto plano)
        extracted_text = document.export_to_markdown()

        # Guarda en un archivo de texto
        with open(output_text_path, "w", encoding="utf-8") as text_file:
            text_file.write(extracted_text)

        print(f"Texto extraído exitosamente a {output_text_path}")
        print("Vista previa del texto extraído:")
        print(extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text)
        
    except Exception as e:
        print(f"Ocurrió un error: {e}")

if __name__ == "__main__":
    extract_text_from_pdf(pdf_path, output_text_path)