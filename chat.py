import httpx

API_URL = "http://localhost:8000/chat"


def main():
    print(
        "Welcome to Gemma Chat! (Type 'quit' or 'exit' to leave, 'clear' to reset history)"
    )
    print("-" * 50)

    history = []

    # Simple check if server is up
    try:
        httpx.get("http://localhost:8000/", timeout=2.0)
    except httpx.RequestError:
        print(
            "Warning: Could not connect to http://localhost:8000. Make sure the server is running via './run.sh'"
        )
        print("-" * 50)

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            if user_input.lower() == "clear":
                history = []
                print("History cleared.")
                continue

            payload = {
                "message": user_input,
                "history": history[-10:],  # Send last 10 messages
                "max_length": 500,
                "temperature": 0.7,
            }

            print("Bot: ...", end="\r", flush=True)

            try:
                # Generous timeout for generation
                response = httpx.post(API_URL, json=payload, timeout=120.0)
                response.raise_for_status()

                print("Bot: ", end="", flush=True)

                data = response.json()
                bot_response = data.get("response", "").strip()

                print(bot_response)

                # Update history
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": bot_response})

                # Keep history manageable locally too
                if len(history) > 20:  # 10 pairs
                    history = history[-20:]

            except httpx.HTTPStatusError as e:
                print(f"\nError: Server returned status {e.response.status_code}")
                try:
                    detail = e.response.json().get("detail", "No detail")
                    print(f"Detail: {detail}")
                except Exception:
                    pass
            except httpx.RequestError as e:
                print(f"\nError: Could not connect to server. ({e})")
            except Exception as e:
                print(f"\nAn error occurred: {e}")

            print("-" * 50)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
