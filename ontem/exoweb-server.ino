#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <ArduinoJson.h>

// Configurações do Access Point
const char* ssid = "EXOWEB";    
const char* password = "12345678";     // Mínimo 8 caracteres

ESP8266WebServer server(80);
float ultimoValorSensor = 0;

void setup() {
Serial.begin(115200); 
delay(1000);

// Configura o Access Point
WiFi.softAPdisconnect(true);
WiFi.mode(WIFI_AP);

bool success = WiFi.softAP(ssid, password);
if(success) {
  Serial.println("\nPonto de Acesso Iniciado com sucesso");
} else {
  Serial.println("\nFalha ao iniciar o Ponto de Acesso");
}

Serial.print("Nome da Rede: ");
Serial.println(ssid);
Serial.print("IP do Servidor: ");
Serial.println(WiFi.softAPIP());

// Configurando rotas
server.on("/", HTTP_GET, handleRoot);  // Rota raiz
server.on("/sensor", HTTP_GET, handleGetSensor);  // Rota GET para ler valor
server.on("/sensor", HTTP_POST, handlePostSensor);  // Rota POST para enviar valor
server.onNotFound(handleNotFound);  // Tratamento de rota não encontrada

server.begin();
Serial.println("Servidor HTTP iniciado");
}

void handleRoot() {
String html = "<html><body>";
html += "<h1>ESP8266 Sensor Server</h1>";
html += "<p>Último valor do sensor: " + String(ultimoValorSensor) + "</p>";
html += "</body></html>";
server.send(200, "text/html", html);
}

void handleGetSensor() {
StaticJsonDocument<200> doc;
doc["valor"] = ultimoValorSensor;
String jsonString;
serializeJson(doc, jsonString);
server.send(200, "application/json", jsonString);
}

void handlePostSensor() {
if (server.hasArg("plain")) {
  String message = server.arg("plain");
  StaticJsonDocument<200> doc;
  DeserializationError error = deserializeJson(doc, message);

  if (!error) {
    ultimoValorSensor = doc["valor"];
    Serial.print("Valor recebido do sensor: ");
    Serial.println(ultimoValorSensor);

    // Responde com o valor recebido
    StaticJsonDocument<200> responseDoc;
    responseDoc["status"] = "success";
    responseDoc["valor"] = ultimoValorSensor;
    String jsonResponse;
    serializeJson(responseDoc, jsonResponse);

    server.send(200, "application/json", jsonResponse);
  } else {
    server.send(400, "application/json", "{\"error\": \"JSON inválido\"}");
  }
} else {
  server.send(400, "application/json", "{\"error\": \"Dados não recebidos\"}");
}
}

void handleNotFound() {
String message = "Rota não encontrada\n\n";
message += "URI: ";
message += server.uri();
message += "\nMethod: ";
message += (server.method() == HTTP_GET) ? "GET" : "POST";
message += "\nArguments: ";
message += server.args();
message += "\n";

for (uint8_t i = 0; i < server.args(); i++) {
  message += " " + server.argName(i) + ": " + server.arg(i) + "\n";
}

server.send(404, "text/plain", message);
}

void loop() {
server.handleClient();
delay(1);
}
