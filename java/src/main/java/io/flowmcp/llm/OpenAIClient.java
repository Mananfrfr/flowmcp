package io.flowmcp.llm;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import java.util.Map;

/**
 * OpenAI LLM client.
 */
public class OpenAIClient implements LLMClient {

    private final String model;
    private final String apiKey;
    private final String baseUrl;
    private final HttpClient httpClient;
    private final Gson gson;

    public OpenAIClient(String model) {
        this(model, System.getenv("OPENAI_API_KEY"), "https://api.openai.com/v1", HttpClient.newHttpClient(), new Gson());
    }

    public OpenAIClient(String model, String apiKey) {
        this(model, apiKey, "https://api.openai.com/v1", HttpClient.newHttpClient(), new Gson());
    }

    public OpenAIClient(String model, String apiKey, String baseUrl, HttpClient httpClient, Gson gson) {
        this.model = model;
        this.apiKey = apiKey;
        this.baseUrl = (baseUrl != null && !baseUrl.isBlank()) ? baseUrl : "https://api.openai.com/v1";
        this.httpClient = httpClient != null ? httpClient : HttpClient.newHttpClient();
        this.gson = gson != null ? gson : new Gson();
    }

    @Override
    public String complete(String prompt) {
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalStateException("OPENAI_API_KEY environment variable is not set.");
        }
        if (model == null || model.isBlank()) {
            throw new IllegalArgumentException("Model cannot be null or empty.");
        }

        try {
            String safePrompt = prompt != null ? prompt : "";
            String body = gson.toJson(Map.of(
                "model", model,
                "messages", List.of(Map.of("role", "user", "content", safePrompt))
            ));

            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/chat/completions"))
                .header("Content-Type", "application/json")
                .header("Authorization", "Bearer " + apiKey)
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new RuntimeException("OpenAI API returned HTTP " + response.statusCode() + ": " + response.body());
            }

            JsonObject root = JsonParser.parseString(response.body()).getAsJsonObject();
            JsonArray choices = root.getAsJsonArray("choices");
            if (choices == null || choices.isEmpty()) {
                throw new RuntimeException("Missing 'choices' in OpenAI response: " + response.body());
            }
            return choices.get(0).getAsJsonObject()
                    .getAsJsonObject("message")
                    .get("content").getAsString();
        } catch (Exception e) {
            throw new RuntimeException("OpenAI request failed: " + e.getMessage(), e);
        }
    }
}
