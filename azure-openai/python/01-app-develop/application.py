import os
import asyncio
from dotenv import load_dotenv

# Add Azure OpenAI package
from openai import AsyncAzureOpenAI


async def main(): 
        
    try: 
    
        # Get configuration settings 
        load_dotenv()
        azure_oai_endpoint = os.getenv("AZURE_OAI_ENDPOINT")
        azure_oai_key = os.getenv("AZURE_OAI_KEY")
        azure_oai_deployment = os.getenv("AZURE_OAI_DEPLOYMENT")
        
        # Configure the Azure OpenAI client
        client = AsyncAzureOpenAI(
            azure_endpoint=azure_oai_endpoint,
            api_key=azure_oai_key,
            api_version="2024-02-15-preview",
        )
        #Initialize messages array
        print("\nAdding grounding context from grounding.txt")
        grounding_text = open(file="grounding.txt", encoding="utf8").read().strip()
        messages_array = [{"role": "user", "content": grounding_text}]

        while True:
            # Pause the app to allow the user to enter the system prompt
            print("------------------\nPausing the app to allow you to change the system prompt.\nPress enter to continue...")
            input()

            # Read in system message and prompt for user message
            system_text = open(file="system.txt", encoding="utf8").read().strip()
            user_text = input("Enter user message, or 'quit' to exit: ")
            if user_text.lower() == 'quit' or system_text.lower() == 'quit':
                print('Exiting program...')
                break

            # Format and send the request to the model
            messages_array.append({"role": "system", "content": system_text})
            messages_array.append({"role": "user", "content": user_text})

            await call_openai_model(system_message = system_text, 
                                    user_message = user_text, 
                                    model=azure_oai_deployment, 
                                    client=client
                                    )

    except Exception as ex:
        print(ex)

# Define the function that will get the response from Azure OpenAI endpoint
async def call_openai_model(system_message, user_message, model, client):
    # Get response from Azure OpenAI
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    print("\nSending request to Azure OpenAI...\n")
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=800,
        temperature=0.7,
    )


    print("Response:\n" + response.choices[0].message.content + "\n")

if __name__ == '__main__': 
    asyncio.run(main())
