import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_url(url: str) -> list[str]:
    """
    Загружает веб-страницу, извлекает из неё текст
    и разбивает текст на небольшие фрагменты.
    """

    response = requests.get(
        url,
        timeout=15,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; RAG-Agent/1.0)"
        },
    )

    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Удаляем элементы, которые обычно не содержат основного текста.
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    text = soup.get_text(separator=" ", strip=True)

    if not text:
        raise ValueError("Не удалось извлечь текст из страницы")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_text(text)

    if not chunks:
        raise ValueError("Не удалось разбить текст страницы на чанки")

    return chunks