def BoxAI(user_input):
    user_input = user_input.lower()
    
    if "hello" in user_input:
        return "Hi there! How can I help you?"
    elif "how are you" in user_input:
        return "I'm just a bot, but I'm doing great! How about you?"
    elif "your name" in user_input:
        return "I'm BoxAI, created to assist you!"
    elif "bye" in user_input:
        return "Goodbye! Have a great day!"
    else:
        return "I'm not sure how to respond to that. Could you ask something else?"

if __name__ == "__main__":
    print("BoxAI: Hello! Type 'bye' to exit.")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "bye":
            print("BoxAI: Goodbye! Have a great day!")
            break
        
        response = BoxAI(user_input)
        print(f"BoxAI: {response}")
