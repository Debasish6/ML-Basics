from configparser import ConfigParser
from chatbot import ChatBot
import sys
import os
from dotenv import load_dotenv

def main():
    # config =ConfigParser()
    # config.read('creadential.ini')
    # api_key = config['GoogleAPIKey']['API_KEY']
    load_dotenv()
    api_key=os.getenv("GoogleAPIKey")

    chatbot = ChatBot(api_key=api_key)
    chatbot.start_conversation()
    print("Welcome to Expand smERP Chat bot. Type 'bye' to exit.")
    # choice =int(input("1.Product Related Questions\n2.Company and ERP Software Related Questions\n"))

    while True:
        user_input = input("You: ")
        if user_input.lower() =='bye':
            response = chatbot.send_prompts(user_input)
            print(f"\n{chatbot.CHATBOT_NAME}: {response}")
            sys.exit("............Exiting ChatBot..........")
        try:
            response = chatbot.send_prompts(user_input)
            print(f"\n{chatbot.CHATBOT_NAME}: {response}")
        except Exception as e:
            print(f'Error: {e}')

if __name__ == "__main__":
    main()