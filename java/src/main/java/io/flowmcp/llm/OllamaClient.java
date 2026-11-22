package io.flowmcp.llm;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.Map;

/**
 * Ollama LLM client — runs local models, no API key needed.
 */
public class OllamaClient implements LLMClient {

    private final String model;
    private final String baseUrl;
    private final HttpClient httpClient;
    private final Gson gson;

    public OllamaClient(String model) {
        this(model, "http://localhost:11434");
    }

    public OllamaClient(String model, String baseUrl) {
        this.model = model;
        this.baseUrl = baseUrl;
        this.httpClient = HttpClient.newHttpClient();
        this.gson = new Gson();
    }

    public OllamaClient(String model, String baseUrl, HttpClient httpClient, Gson gson) {
        this.model = model;
        this.baseUrl = baseUrl;
        this.httpClient = httpClient;
        this.gson = gson != null ? gson : new Gson();
    }

    @Override
    public String complete(String prompt) {
        try {
            String body = gson.toJson(Map.of(
                "model", model,
                "prompt", prompt != null ? prompt : "",
                "stream", false
            ));

            HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/api/generate"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(body))
                .build();

            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                throw new RuntimeException("Ollama API returned HTTP status " + response.statusCode() + ": " + response.body());
            }

            JsonObject root = JsonParser.parseString(response.body()).getAsJsonObject();
            if (!root.has("response") || root.get("response").isJsonNull()) {
                throw new RuntimeException("Missing 'response' field in Ollama response: " + response.body());
            }
            return root.get("response").getAsString();
        } catch (Exception e) {
            throw new RuntimeException("Ollama request failed: " + e.getMessage(), e);
        }
    }
}
