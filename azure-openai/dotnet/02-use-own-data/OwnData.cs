using System.ClientModel;
using Microsoft.Extensions.Configuration;

// Add Azure OpenAI package
using Azure.AI.OpenAI;
using Azure.Search.Documents.Models;
using OpenAI.Chat;
using Azure.AI.OpenAI.Chat;
// Add Azure Search package
using Azure.AI.OpenAI.DataSources;

// Flag to show citations
bool showCitations = false;

// Get configuration settings  
IConfiguration config = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json")
    .Build();
string oaiEndpoint = config["AzureOAIEndpoint"] ?? "";
string oaiKey = config["AzureOAIKey"] ?? "";
string oaiDeploymentName = config["AzureOAIDeploymentName"] ?? "";
string azureSearchEndpoint = config["AzureSearchEndpoint"] ?? "";
string azureSearchKey = config["AzureSearchKey"] ?? "";
string azureSearchIndex = config["AzureSearchIndex"] ?? "";

// Initialize the Azure OpenAI client
AzureOpenAIClient azureClient = new (new Uri(oaiEndpoint), new ApiKeyCredential(oaiKey));
ChatClient chatClient = azureClient.GetChatClient(oaiDeploymentName);

// Get the prompt text
Console.WriteLine("Enter a question:");
string text = Console.ReadLine() ?? "";

// Configure your data source
#pragma warning disable AOAI001
ChatCompletionsOptions chatCompletionsOptions = new ChatCompletionsOptions
{
    // Set the temperature for the model response
    Temperature = 0.9f,
    // Set the maximum number of tokens to generate in the response
    MaxOutputTokenCount = 600,
};

chatCompletionsOptions.AddUserDataSource(
    new AzureSearchDataSource
    {
        Endpoint = new Uri(azureSearchEndpoint),
        IndexName = azureSearchIndex,
        Authentication = DataSourceAuthentication.FromApiKey(azureSearchKey)
    });
// Send request to Azure OpenAI model  
Console.WriteLine("...Sending the following request to Azure OpenAI endpoint...");  
Console.WriteLine("Request: " + text + "\n");

ChatCompletion completion = chatClient.CompleteChat(
    [
        new SystemChatMessage("You are an AI assistant that helps with travel-related inquiries, offering tips, advice, and recommendations as a knowledgeable travel agent."),
        new UserChatMessage(text),
    ],
    chatCompletionsOptions);

ChatMessageContext onYourDataContext = completion.GetMessageContext();

if (onYourDataContext?.Intent is not null)
{
    Console.WriteLine($"Intent: {onYourDataContext.Intent}");
}

// Print response
Console.WriteLine($"{completion.Role}: {completion.Content[0].Text}");

if (showCitations)
{
    Console.WriteLine($"\n  Citations of data used:");

    foreach (ChatCitation citation in onYourDataContext?.Citations ?? [])
    {
        Console.WriteLine($"Citation: {citation.Content}");
    }
}
