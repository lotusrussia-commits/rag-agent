import requests
from langchain_core.tools import tool


@tool
def get_cat_fact() -> str:
    """
    Получает случайный факт о кошках через открытый API.
    """

    url = "https://catfact.ninja/fact"

    try:
        response = requests.get(
            url,
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        fact = data.get("fact")

        if not fact:
            return "API не вернул текст факта."

        return fact

    except requests.RequestException as error:
        return f"Ошибка при обращении к API: {error}"

    except ValueError:
        return "API вернул некорректный JSON."