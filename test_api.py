from app.tools import get_cat_fact


def main():
    print("Запрашиваем факт о кошках через открытый API...")
    print()

    result = get_cat_fact.invoke({})

    print("Ответ API:")
    print(result)


if __name__ == "__main__":
    main()