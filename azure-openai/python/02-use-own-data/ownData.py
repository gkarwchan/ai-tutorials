import os
import openai
import dotenv

## Flag to show citations
showCitations = False

dotenv.load_dotenv()

endpoint = os.environ.get("AZURE_OAI_ENDPOINT")
api_key = os.environ.get("AZURE_OAI_KEY")
deployment = os.environ.get("AZURE_OAI_DEPLOYMENT")

client = openai.AzureOpenAI(
    azure_endpoint=endpoint,
    api_key=api_key,
    api_version="2024-02-01",
)

# Configure your data source
text = input("Enter your question:\n")
completion = client.chat.completions.create(
    model=deployment,
    messages=[
        {
            "role": "user",
            "content": text
        }
    ],
    extra_body={
        "data_sources":[
            {
                "type": "azure_search",
                "parameters": {
                    "endpoint": os.environ.get("AZURE_SEARCH_ENDPOINT"),
                    "key": os.environ.get("AZURE_SEARCH_KEY"),
                    "index_name": os.environ.get("AZURE_SEARCH_INDEX"),
                    "authentication": {
                        "type": "api_key",
                        "key": os.environ.get("AZURE_SEARCH_KEY")
                    }
                }
            }
        ],
    }
)
print(completion.choices[0].message.content)

if showCitations:
    print(f"\n{completion.choices[0].message.context}")
