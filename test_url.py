from app.rag_agent import RAGAgent


def main():
    url = "https://example.com"

    agent = RAGAgent()

    print(f"Загружаем страницу: {url}")

    chunks_count = agent.add_url(url)

    print(f"Добавлено чанков в Pinecone: {chunks_count}")
    print()

    query = "Для чего используется этот домен?"

    print(f"Поисковый запрос: {query}")
    print()

    results = agent.search(query, k=3)

    print("Найденные фрагменты:")
    print()

    for i, document in enumerate(results, start=1):
        print(f"--- Результат {i} ---")
        print(document.page_content)
        print(f"Источник: {document.metadata.get('source')}")
        print()


if __name__ == "__main__":
    main()