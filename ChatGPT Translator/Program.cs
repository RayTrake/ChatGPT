using OpenAI.GPT3;
using OpenAI.GPT3.Managers;
using OpenAI.GPT3.ObjectModels;
using OpenAI.GPT3.ObjectModels.RequestModels;
using ServiceStack;
using System.Diagnostics;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json.Serialization;

public class AppConfiguration
{
    public string ApiKey { get; set; }
    public string DesiredLanguage { get; set; }
    public int StartFromLine { get; set; }
}

class Program
{
    public static AppConfiguration Configuration { get; set; }

    #region Legacy
    static string endpoint = "https://api.openai.com/v1/chat/completions";
    static HttpClient httpClient = new HttpClient();

    public static class Role
    {
        public static string System => "system";
        public static string User => "user";
        public static string Assistant => "assistant";
    }

    public static class Model
    {
        public static string Gpt4 => "gpt-4";
        public static string Gpt35 => "gpt-3.5-turbo";
    }

    public static void InitHttp()
    {
        httpClient.DefaultRequestHeaders.Add("Authorization", $"Bearer {Configuration.ApiKey}");
    }

    public static async Task<string> SendRequest(string request, bool useGpt4 = true)
    {
        string modelId = useGpt4 ? Model.Gpt4 : Model.Gpt35;

        Message msg = new Message()
        {
            Role = Role.User,
            Content = request
        };

        var requestData = new Request()
        {
            ModelId = modelId,
            Messages = new List<Message> { msg }
        };

        using var response = await httpClient.PostAsJsonAsync(endpoint, requestData);

        if (!response.IsSuccessStatusCode)
        {
            Console.WriteLine($"{(int)response.StatusCode} {response.StatusCode}");
            return "-1"; // error code
        }

        ResponseData? responseData = await response.Content.ReadFromJsonAsync<ResponseData>();

        var choices = responseData?.Choices ?? new List<Choice>();
        if (choices.Count == 0)
        {
            Console.WriteLine("No choices were returned by the API");
            return "-1";
        }

        var choice = choices[0];
        var responseMessage = choice.Message;

        var responseText = responseMessage.Content.Trim();
        return responseText;
    }

    class Message
    {
        [JsonPropertyName("role")]
        public string Role { get; set; } = "";
        [JsonPropertyName("content")]
        public string Content { get; set; } = "";
    }

    class Request
    {
        [JsonPropertyName("model")]
        public string ModelId { get; set; } = "";
        [JsonPropertyName("messages")]
        public List<Message> Messages { get; set; } = new();
    }

    class ResponseData
    {
        [JsonPropertyName("id")]
        public string Id { get; set; } = "";
        [JsonPropertyName("object")]
        public string Object { get; set; } = "";
        [JsonPropertyName("created")]
        public ulong Created { get; set; }
        [JsonPropertyName("choices")]
        public List<Choice> Choices { get; set; } = new();
        [JsonPropertyName("usage")]
        public Usage Usage { get; set; } = new();
    }

    class Choice
    {
        [JsonPropertyName("index")]
        public int Index { get; set; }
        [JsonPropertyName("message")]
        public Message Message { get; set; } = new();
        [JsonPropertyName("finish_reason")]
        public string FinishReason { get; set; } = "";
    }

    class Usage
    {
        [JsonPropertyName("prompt_tokens")]
        public int PromptTokens { get; set; }
        [JsonPropertyName("completion_tokens")]
        public int CompletionTokens { get; set; }
        [JsonPropertyName("total_tokens")]
        public int TotalTokens { get; set; }
    }

    #endregion

    public static Dictionary<string, string[]> Data = new Dictionary<string, string[]>();
    private static void LoadSourceLanguage()
    {
        Data.Clear();
        string filePath = Path.Combine("English.lng");
        if (File.Exists(filePath))
        {
            var languageData = File.ReadAllLines(filePath);
            foreach (var d in languageData)
            {
                string[] opts = d.Split(';');
                string key = opts[0];
                string[] langData = new string[opts.Length - 1];

                for (int i = 1; i < opts.Length; i++)
                {
                    langData[i - 1] = opts[i].Replace("\\n", "\n").Replace("◙", ";");
                }

                Data.Add(key, langData);
            }
        }
    }

    static OpenAIService gpt;

    static void InitGPT()
    {
        gpt = new OpenAIService(new OpenAiOptions()
        {
            ApiKey = Configuration.ApiKey
        });
    }

    static void ReadConfig()
    {
        string configText = File.ReadAllText("Application.configuration");
        Configuration = configText.FromJson<AppConfiguration>();
    }

    static async Task Main(string[] _)
    {
        Console.OutputEncoding = Encoding.UTF8;
        ReadConfig();
        InitGPT();
        LoadSourceLanguage();

        Data = Data.Skip(Configuration.StartFromLine).ToDictionary(x => x.Key, x => x.Value);

        int processed = 0;
        int totalAmount = Data.Count;

        Stopwatch sw = new Stopwatch();
        sw.Start();

        foreach (var d in Data)
        {
            Console.Title = $"Process {processed}/{totalAmount}, Elapsed: {sw.ElapsedMilliseconds / 1000} seconds";

            string textToTranslate = d.Value[0];
            string additionalInfo = d.Value.Length > 1 ? d.Value[1] : string.Empty;
            string result = "";

            Console.WriteLine($"\nSource: {textToTranslate}");

            int emptyCounter = 0;
            do
            {
                result = await Translate(textToTranslate, Configuration.DesiredLanguage, additionalInfo);
                if (string.IsNullOrEmpty(result))
                {
                    result = "[EMPTY]";
                    emptyCounter++;
                    Thread.Sleep(1000);
                }

                if (emptyCounter >= 3)
                {
                    break;
                }
            }
            while (string.IsNullOrEmpty(result));
            processed++;

            Thread.Sleep(100);
            Console.WriteLine($"Result: {result}");

            AppendLineToFile($"{Configuration.DesiredLanguage}.lng", d.Key, result);

            // clear console every 50 keys
            if (processed % 50 == 0)
            {
                Console.Clear();
            }
        }

        Console.WriteLine("Done");
        Console.ReadKey();
    }

    public static async Task<string> Translate(string text, string language, string additionalInfo)
    {
        try
        {
            var result = await gpt.ChatCompletion.CreateCompletion(new ChatCompletionCreateRequest()
            {
                Model = Models.Gpt_4,
                Messages =
                new List<ChatMessage>()
                {
                    new ChatMessage(StaticValues.ChatMessageRoles.User, $"please translate text below to {Configuration.DesiredLanguage} language, do not translate words with curly or square brackets, do not add any notes, give me only text translation, {additionalInfo}:\n{text}"),
                },
                Temperature = 0.5f
            });

            Console.OutputEncoding = Encoding.UTF8;
            if (result.Successful)
            {
                string content = result.Choices.First().Message.Content;
                content = content.Trim();

                result = null;
                return content;
            }
            else
            {
                if (result.Error == null)
                {
                    throw new Exception("Unknown Error");
                }

                Console.WriteLine($"{result.Error.Code}: {result.Error.Message}");
            }
        }
        catch
        {
        }

        return string.Empty;
    }

    private static List<string> result = new List<string>();
    public static void AppendLineToFile(string file, string key, string translation)
    {
        // write to file as separate line
        result.Clear();
        result.Add($"{key};{translation.Replace("\r\n", "\\n").Replace("\n", "\\n")}");
        System.IO.File.AppendAllLines(file, result);
    }
}