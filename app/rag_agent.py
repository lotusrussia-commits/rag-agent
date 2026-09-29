import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.url_loader import load_url


load_dotenv()


class RAGAgent:
    def __init__(self):
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.namespace = os.getenv("PINECONE_NAMESPACE", "default")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")

        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME не найден в .env")

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY не найден в .env")

        embedding_kwargs = {
            "model": "text-embedding-3-small",
            "api_key": self.openai_api_key,
        }

        if self.openai_base_url:
            embedding_kwargs["base_url"] = self.openai_base_url

        self.embeddings = OpenAIEmbeddings(**embedding_kwargs)

        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings,
            namespace=self.namespace,
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self.openai_api_key,
            base_url=self.openai_base_url,
            temperature=0,
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Ты RAG-ассистент.

Отвечай на вопрос пользователя только на основе
переданного контекста.

Если в контексте нет информации для ответа,
честно скажи, что информации недостаточно.

Не придумывай факты.

Отвечай кратко и по существу.""",
                ),
                (
                    "human",
                    """Контекст:

{context}

Вопрос пользователя:

{question}

Ответ:""",
                ),
            ]
        )

    def add_documents(self, documents: list[str]) -> None:
        self.vector_store.add_texts(documents)

    def add_url(self, url: str) -> int:
        """
        Загружает URL, разбивает страницу на чанки
        и сохраняет чанки в Pinecone.
        """

        chunks = load_url(url)

        self.vector_store.add_texts(
            chunks,
            metadatas=[
                {"source": url}
                for _ in chunks
            ],
        )

        return len(chunks)

    def search(self, query: str, k: int = 3):
        return self.vector_store.similarity_search(query, k=k)

    def answer(self, question: str, k: int = 3) -> str:
        """
        Ищет релевантные фрагменты в Pinecone
        и формирует ответ через LLM.
        """

        documents = self.search(question, k=k)

        if not documents:
            return "Не удалось найти релевантную информацию."

        context_parts = []

        for document in documents:
            source = document.metadata.get("source", "неизвестный источник")

            context_parts.append(
                f"Источник: {source}\n"
                f"Текст: {document.page_content}"
            )

        context = "\n\n".join(context_parts)

        messages = self.prompt.format_messages(
            context=context,
            question=question,
        )

        response = self.llm.invoke(messages)

        return response.content