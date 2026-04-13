import random


def main() -> None:
    target = random.randint(0, 100)
    tries = 0

    print("🎯 Guess the Number: Gradient Descent Edition")
    print("I'm thinking of a number between 0 and 100.")
    print("Each hint nudges you up ⬆️ or down ⬇️, like gradient descent steps!")
    print("Type 'q' to quit.\n")

    while True:
        user_input = input("Your guess (0-100): ").strip().lower()

        if user_input == "q":
            print(f"👋 Game over. The number was {target}.")
            break

        if not user_input.isdigit():
            print("⚠️ Please enter a whole number from 0 to 100 (or 'q').")
            continue

        guess = int(user_input)
        if guess < 0 or guess > 100:
            print("🚧 Stay in range: 0 to 100.")
            continue

        tries += 1
        if guess < target:
            print("⬆️ Too low! Move up.")
        elif guess > target:
            print("⬇️ Too high! Move down.")
        else:
            print(f"✅ Nice! You found it in {tries} tries.")
            if tries <= 5:
                print("🚀 Super efficient descent!")
            elif tries <= 10:
                print("😎 Good convergence.")
            else:
                print("🐢 Slow but steady convergence.")
            print("🧠 Gradient descent idea: each hint points you toward the minimum error.")
            break


if __name__ == "__main__":
    main()
