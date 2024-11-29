#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ArduinoJson.h>
#include <LittleFS.h>

// Configurações do Access Point
const char* ssid = "EXOWEB";    
const char* password = "12345678";

ESP8266WebServer server(80);
const char* ARQUIVO_DADOS = "/sensores.json";
const int MAX_REGISTROS = 100;

// Estrutura para armazenar dados do sensor
struct DadoSensor {
  float valor;
  String tipo;
  unsigned long timestamp;
};

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Inicializa o sistema de arquivos
  if (!LittleFS.begin()) {
    Serial.println("Erro ao montar LittleFS");
    return;
  }

  // Configura o Access Point
  WiFi.softAPdisconnect(true);
  WiFi.mode(WIFI_AP);
  
  bool success = WiFi.softAP(ssid, password);
  if(success) {
    Serial.println("\nPonto de Acesso Iniciado com sucesso");
  } else {
    Serial.println("\nFalha ao iniciar o Ponto de Acesso");
  }

  Serial.print("IP do Servidor: ");
  Serial.println(WiFi.softAPIP());

  // Configurando rotas
  server.on("/", HTTP_GET, handleRoot);
  server.on("/sensor", HTTP_POST, handlePostSensor);
  server.on("/sensores", HTTP_GET, handleGetSensores);
  server.on("/ultimo", HTTP_GET, handleGetUltimo);
  server.onNotFound(handleNotFound);

  server.begin();
  Serial.println("Servidor HTTP iniciado");
}

void handleRoot() {
  String html = "<html><body>";
  html += "<h1>ESP8266 Sensor Server</h1>";
  html += "<p>Endpoints disponíveis:</p>";
  html += "<ul>";
  html += "<li>POST /sensor - Enviar dados do sensor</li>";
  html += "<li>GET /sensores - Listar todos os dados</li>";
  html += "<li>GET /ultimo - Último valor registrado</li>";
  html += "</ul>";
  html += "</body></html>";
  server.send(200, "text/html", html);
}

void handlePostSensor() {
  if (server.hasArg("plain")) {
    String message = server.arg("plain");
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, message);

    if (!error) {
      // Lê os dados existentes
      DynamicJsonDocument dados(16384); // 16KB para dados
      lerDadosArquivo(dados);
      
      // Adiciona novo registro
      JsonArray array = dados["sensores"].as<JsonArray>();
      JsonObject novoRegistro = array.createNestedObject();
      
      novoRegistro["valor"] = doc["valor"].as<float>();
      novoRegistro["tipo"] = doc["tipo"] | "generico";
      novoRegistro["timestamp"] = millis();

      // Mantém apenas os últimos MAX_REGISTROS
      while (array.size() > MAX_REGISTROS) {
        array.remove(0);
      }

      // Salva os dados atualizados
      if (salvarDadosArquivo(dados)) {
        server.send(200, "application/json", "{\"status\": \"success\", \"message\": \"Dados salvos\"}");
      } else {
        server.send(500, "application/json", "{\"error\": \"Erro ao salvar dados\"}");
      }
    } else {
      server.send(400, "application/json", "{\"error\": \"JSON inválido\"}");
    }
  } else {
    server.send(400, "application/json", "{\"error\": \"Dados não recebidos\"}");
  }
}

void handleGetSensores() {
  DynamicJsonDocument dados(16384);
  if (lerDadosArquivo(dados)) {
    String response;
    serializeJson(dados, response);
    server.send(200, "application/json", response);
  } else {
    server.send(500, "application/json", "{\"error\": \"Erro ao ler dados\"}");
  }
}

void handleGetUltimo() {
  DynamicJsonDocument dados(16384);
  if (lerDadosArquivo(dados)) {
    JsonArray array = dados["sensores"].as<JsonArray>();
    if (array.size() > 0) {
      String response;
      serializeJson(array[array.size() - 1], response);
      server.send(200, "application/json", response);
    } else {
      server.send(404, "application/json", "{\"error\": \"Nenhum dado encontrado\"}");
    }
  } else {
    server.send(500, "application/json", "{\"error\": \"Erro ao ler dados\"}");
  }
}

bool lerDadosArquivo(JsonDocument& dados) {
  if (!LittleFS.exists(ARQUIVO_DADOS)) {
    dados.clear();
    dados.createNestedArray("sensores");
    return true;
  }

  File file = LittleFS.open(ARQUIVO_DADOS, "r");
  if (!file) {
    return false;
  }

  DeserializationError error = deserializeJson(dados, file);
  file.close();

  if (error) {
    return false;
  }

  return true;
}

bool salvarDadosArquivo(const JsonDocument& dados) {
  File file = LittleFS.open(ARQUIVO_DADOS, "w");
  if (!file) {
    return false;
  }

  serializeJson(dados, file);
  file.close();
  return true;
}

void handleNotFound() {
  String message = "Rota não encontrada\n\n";
  message += "URI: ";
  message += server.uri();
  message += "\nMethod: ";
  message += (server.method() == HTTP_GET) ? "GET" : "POST";
  server.send(404, "text/plain", message);
}

void loop() {
  server.handleClient();
  delay(1);
}
