"""Ask questions grounded in your study materials via retrieval + Groq.

Run from the project root:
    python -m scripts.ask
"""

from src.generate import GenerationError, MissingAPIKeyError, generate_answer


def main():
    print("Ask a question about your study materials (empty line to quit).\n")
    while True:
        question = input("ask> ").strip()
        if not question:
            break

        try:
            result = generate_answer(question)
        except MissingAPIKeyError as e:
            print(f"\n{e}\n")
            break
        except GenerationError as e:
            print(f"\n{e}\n")
            continue

        print(f"\n{result['answer']}\n")
        if result["sources"]:
            cited = ", ".join(f"{s['source']}#{s['chunk_index']}" for s in result["sources"])
            print(f"Sources: {cited}")
        print()


if __name__ == "__main__":
    main()
