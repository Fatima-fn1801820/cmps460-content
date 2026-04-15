import random


def main() -> None:
    target = random.randint(0, 100)
    tries = 0

    print("🎯 Guess the Number: Gradient Descent Edition")
    print("I'm thinking of a number between 0 and 100.")
    print("Each hint nudges you up ⬆️ or down ⬇️, like gradient descent steps!")
    print("Hint strength guides step size:")
    print("- too low / too high -> take a bigger step")
    print("- low / high         -> take a smaller step")
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
        diff = guess - target
        distance = abs(diff)

        if diff < 0:
            if distance >= 20:
                print("⬆️ Too low! You're far below the target - move up more aggressively.")
            else:
                print("↗️ Low. You're close but still below the target - move up a little.")
        elif diff > 0:
            if distance >= 20:
                print("⬇️ Too high! You're far above the target - move down more aggressively.")
            else:
                print("↘️ High. You're close but still above the target - move down a little.")
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
