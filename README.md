# RAG-агент

Агент загружает веб-страницу, сохраняет её фрагменты в векторную базу Pinecone и отвечает на вопросы только по найденному контексту. Ответы формирует `gpt-4o-mini` через [ProxyAPI](https://proxyapi.ru/). Отдельно проект обращается к публичному API [catfact.ninja](https://catfact.ninja/fact) и возвращает случайный факт о кошках.

## Что делает проект

1. Скачивает HTML по URL и извлекает из него текст.
2. Режет текст на перекрывающиеся фрагменты.
3. Считает эмбеддинги моделью `text-embedding-3-small` и записывает векторы в Pinecone вместе с адресом страницы.
4. По вопросу пользователя ищет три ближайших фрагмента.
5. Передаёт их в промпт и получает краткий ответ от `gpt-4o-mini`.
6. Если подходящих фрагментов нет, сообщает, что информации недостаточно, и не достраивает факты.

Точка входа — `main.py`. Она загружает страницу брендов Volkswagen Group и задаёт вопрос о группах Core, Progressive и Sport Luxury.

## Архитектура RAG

```text
URL
  → HTTP-запрос и очистка HTML          app/url_loader.py
  → чанки по 1000 символов, overlap 200
  → эмбеддинг text-embedding-3-small
  → Pinecone (индекс + namespace, metadata source)

вопрос
  → similarity search, k = 3
  → промпт: контекст + вопрос
  → gpt-4o-mini через ProxyAPI
  → ответ
```

Класс `RAGAgent` в `app/rag_agent.py` связывает эти шаги.

| Метод | Назначение |
| --- | --- |
| `add_url` | Загрузка страницы, нарезка, запись чанков в Pinecone |
| `add_documents` | Запись уже готовых текстов |
| `search` | Поиск ближайших фрагментов |
| `answer` | Поиск и генерация ответа по найденному контексту |

В системном промпте зафиксированы три правила: отвечать только по переданному контексту, прямо говорить, если данных не хватает, и не придумывать факты. Температура модели равна `0`.

Каждый найденный фрагмент попадает в промпт вместе с полем `source` из метаданных, поэтому в контексте виден URL исходной страницы.

## Pinecone и text-embedding-3-small

Эмбеддинги считает `OpenAIEmbeddings` с моделью `text-embedding-3-small`. Векторы хранит `PineconeVectorStore`.

Индекс в проекте не создаётся: он должен уже существовать в Pinecone. Для `text-embedding-3-small` размерность индекса — **1536**.

В `RAGAgent` из окружения читаются `PINECONE_INDEX_NAME` и `PINECONE_NAMESPACE`:

| Переменная | Назначение |
| --- | --- |
| `PINECONE_API_KEY` | Ключ Pinecone. Есть в `.env.example`. В коде `app/` переменная не читается |
| `PINECONE_INDEX_NAME` | Имя индекса. Читается через `os.getenv`. В примере — `car-memory-homework` |
| `PINECONE_NAMESPACE` | Пространство имён внутри индекса. Читается через `os.getenv`; если переменной нет, используется `default` |

У каждого чанка в метаданных сохраняется `source` — URL страницы, из которой он получен.

Повторный запуск `main.py` снова добавляет чанки той же страницы в тот же namespace. Для чистого прогона перед демонстрацией очистите namespace в консоли Pinecone.

## Обработка URL

`load_url` в `app/url_loader.py`:

1. Запрашивает страницу через `requests` с таймаутом 15 секунд и заголовком `User-Agent`.
2. Разбирает HTML через BeautifulSoup.
3. Удаляет `script`, `style` и `noscript`.
4. Собирает видимый текст.
5. Делит его `RecursiveCharacterTextSplitter`: `chunk_size=1000`, `chunk_overlap=200`.

Пустая страница и пустой результат нарезки приводят к ошибке, а не к записи пустых векторов. `add_url` возвращает число сохранённых чанков.

Проверка загрузки и поиска без генерации ответа:

```bash
python test_url.py
```

Скрипт индексирует `https://example.com` и печатает три ближайших фрагмента по запросу «Для чего используется этот домен?».

## Реальный API catfact.ninja

Инструмент `get_cat_fact` в `app/tools.py` вызывает публичный endpoint:

```text
GET https://catfact.ninja/fact
```

Ответ — JSON с полем `fact`. Инструмент оформлен как LangChain `@tool`: таймаут 10 секунд, проверка HTTP-статуса и разбор JSON. При сетевой ошибке, невалидном JSON или пустом поле `fact` возвращается текст ошибки.

Запуск:

```bash
python test_api.py
```

Этот вызов живёт отдельно от ответа по документам: `RAGAgent.answer` ищет контекст в Pinecone и не обращается к catfact.ninja.

## GPT-4o-mini через ProxyAPI

Чат-модель — `ChatOpenAI` с именем `gpt-4o-mini`. Запросы к OpenAI идут не напрямую, а через ProxyAPI: базовый URL задаётся в `OPENAI_BASE_URL`.

| Переменная | Назначение |
| --- | --- |
| `OPENAI_API_KEY` | Ключ ProxyAPI |
| `OPENAI_BASE_URL` | `https://api.proxyapi.ru/v1` |

Если не задан `PINECONE_INDEX_NAME` или `OPENAI_API_KEY`, создание агента останавливается с `ValueError`: в сообщении указано, какой переменной не хватает. `OPENAI_BASE_URL` необязателен. Когда он задан, этот адрес передаётся и в эмбеддинги, и в `ChatOpenAI`.

## Пример: страница Volkswagen Group

`main.py` загружает реальную страницу брендов [Volkswagen Group](https://www.volkswagen-group.com/en/brands-and-brand-groups-15811).

После индексации агент отвечает на вопрос:

> Какие бренды входят в Volkswagen Group и как они распределены по группам Core, Progressive и Sport Luxury?

Запуск:

```bash
python main.py
```

В консоли будет адрес страницы, число добавленных чанков, вопрос и ответ. Ответ строится по трём фрагментам, которые поиск счёл ближайшими к вопросу.

## Установка и запуск

Нужны Python 3.12, индекс Pinecone размерности 1536 и ключи Pinecone и ProxyAPI.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Заполните `.env` своими ключами. Файл `.env` в git не попадает.

```bash
python main.py       # загрузка страницы Volkswagen Group и ответ
python test_url.py   # загрузка example.com и поиск фрагментов
python test_api.py   # факт о кошках с catfact.ninja
```

Зависимости: `langchain`, `langchain-openai`, `langchain-pinecone`, `langchain-community`, `langchain-text-splitters`, `pinecone`, `python-dotenv`, `requests`, `beautifulsoup4`.

## Структура проекта

```text
rag-agent/
├── main.py              # демо: Volkswagen Group → Pinecone → ответ
├── test_url.py          # демо загрузки URL и поиска
├── test_api.py          # демо вызова catfact.ninja
├── requirements.txt
├── .env.example
└── app/
    ├── rag_agent.py     # эмбеддинги, Pinecone, промпт, ответ
    ├── url_loader.py    # скачивание страницы и нарезка на чанки
    └── tools.py         # инструмент get_cat_fact
```

## Соответствие трём пунктам домашнего задания

| Пункт | Требование | Где выполнено |
| --- | --- | --- |
| 1 | Базовый RAG-агент с векторным поиском | `text-embedding-3-small` и `PineconeVectorStore` в `app/rag_agent.py`. Поиск — `similarity_search`, ответ строится по найденным чанкам |
| 2 | Загрузка знаний из внешнего URL | `app/url_loader.py` и `RAGAgent.add_url`. Рабочий пример — страница брендов Volkswagen Group в `main.py` |
| 3 | Вызов реального внешнего API | `get_cat_fact` в `app/tools.py` ходит в `https://catfact.ninja/fact`. Проверка — `python test_api.py` |

Генерация ответа закрывает модельную часть задания: `gpt-4o-mini` вызывается через ProxyAPI (`OPENAI_BASE_URL=https://api.proxyapi.ru/v1`), с температурой `0` и ограничением «только переданный контекст».
