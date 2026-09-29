from app.rag_agent import RAGAgent


def main():
    agent = RAGAgent()

    url = "https://www.volkswagen-group.com/en/brands-and-brand-groups-15811"

    print(f"Загружаем страницу: {url}")
    chunks_count = agent.add_url(url)

    print(f"Добавлено чанков: {chunks_count}")
    print()

    question = (
        "Какие бренды входят в Volkswagen Group и как они распределены "
        "по группам Core, Progressive и Sport Luxury?"
    )

    print(f"Вопрос: {question}")
    print()

    answer = agent.answer(question)

    print("Ответ RAG-агента:")
    print(answer)


if __name__ == "__main__":
    main()