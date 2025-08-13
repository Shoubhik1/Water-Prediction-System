#include <Arduino.h>
#include <Wire.h>
#include <EEPROM.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <SH1106Wire.h>  // SH1106 OLED Display Library
#include <WiFiClientSecure.h>

// OLED Display (SH1106) - Correct I2C Pins
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
SH1106Wire display(0x3C, 26, 25);  // Address, SDA (GPIO 26), SCL (GPIO 25)

// DS18B20 Temperature Sensor Setup
#define ONE_WIRE_BUS 14  // GPIO 14 for DS18B20
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature sensors(&oneWire);

// TDS (EC) Sensor
#define TDS_PIN 36       // TDS Sensor on GPIO 36 (VP)

// WiFi & ThingSpeak Configuration
String apiKey = "S15GM7SY3YD1LXUB";  // Replace with your ThingSpeak API key
const char *ssid = "Alpha";  // WiFi SSID
const char *pass = "74314420";  // WiFi Password
const char *server = "api.thingspeak.com";
WiFiClient client;

// Google Sheets Configuration
const char *googleScriptURL = "https://script.google.com/macros/s/AKfycby6E0Ujip6JU7hWX9OI9Zk6i6mpMfo87KikqbTV1LSuCZUEFXDjh36WPslCOrfBmV6_IQ/exec"; // Replace with your Google Apps Script URL

// Sensor Data Variables
float voltage, tdsValue, ecValue, tdsFactor, temperature = 25;
String turbidityData = ""; 
float turbidityVal = 0.0; 
float turbidityValue = 0.0; // Turbidity value

// Calibration Constants
#define VREF 3.3       // ESP32 reference voltage (3.3V)
#define ADC_MAX 4095   // ESP32 ADC resolution (12-bit)

// Safety Thresholds
#define EC_THRESHOLD 0.5     // Max EC for drinkable water (mS/cm)
#define PPM_THRESHOLD 350
#define TURB_THRESHOLD 5
bool showWaterStatus = false; // Toggle variable

void setup() {
  Serial.begin(115200);  
  Serial2.begin(115200, SERIAL_8N1, 16, 17);  // Arduino (RX: GPIO 16, TX: GPIO 17)

  sensors.begin();

  // Initialize OLED Display
  display.init();
  display.flipScreenVertically();
  display.clear();

  analogReadResolution(12);
  analogSetPinAttenuation(TDS_PIN, ADC_11db);

  // WiFi Connection
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print("...");
  }
  Serial.println("\nWiFi Connected!");
}

void loop() {
  // Read Temperature
  sensors.requestTemperatures();
  temperature = sensors.getTempCByIndex(0);

  // Read TDS (EC) Sensor
  int rawADC = analogRead(TDS_PIN);
  voltage = (rawADC / (float)ADC_MAX) * VREF;  
  float compensationFactor = 1.0 + 0.018 * (temperature - 25.0);
  float compensatedVoltage = voltage / compensationFactor;
  
  if (compensatedVoltage < 1.30) {
    tdsFactor = 0.4;
  } else {
    tdsFactor = 1.0;
  }

  // TDS Calculation
  tdsValue = (rawADC / (float)ADC_MAX) * 1000 * tdsFactor;  
  ecValue = (tdsValue / 0.7) / 1000;

  // Read Turbidity Sensor Data from Arduino via Serial2
  if (Serial2.available()) {
    turbidityData = Serial2.readStringUntil('\n');  
    turbidityVal = turbidityData.toFloat(); 
    turbidityValue = turbidityVal / 1.0; 
  }

  // Debugging Output
  Serial.println("====================================");
  Serial.print("Temperature: "); Serial.print(temperature, 2); Serial.println("°C");
  Serial.print("TDS: "); Serial.print(tdsValue, 2); Serial.println(" ppm");
  Serial.print("EC: "); Serial.print(ecValue, 4); Serial.println(" mS/cm");
  Serial.print("Turb: "); Serial.println(turbidityValue);
  Serial.println(compensatedVoltage);
  Serial.println("====================================");

  // Check Water Quality
  bool isDrinkable = (ecValue <= EC_THRESHOLD) && (turbidityValue <= TURB_THRESHOLD) && (tdsValue <= PPM_THRESHOLD);

  // Toggle OLED display
  display.clear();
  display.setFont(ArialMT_Plain_16);

  if (showWaterStatus) {
    display.drawString(0, 0, "Water Status:");
    if (isDrinkable) {
      display.drawString(0, 20, "✅ SAFE to drink");
    } else {
      display.drawString(0, 20, "⚠️ UNSAFE!");
    }
  } else {
    display.drawString(0, 0, "T: " + String(temperature, 2) + " C");
    display.drawString(0, 16, "TDS: " + String(tdsValue, 2) + " ppm");
    display.drawString(0, 32, "EC: " + String(ecValue, 4) + " mS/cm");
    display.drawString(0, 48, "Turb: " + String(turbidityValue) + " NTU");
  }

  display.display();
  delay(2000);
  showWaterStatus = !showWaterStatus;

  // Send Data
  sendToThingSpeak(temperature, tdsValue, ecValue, turbidityValue, isDrinkable);
  sendToGoogleSheets(temperature, tdsValue, ecValue, turbidityValue, isDrinkable);
}

// Function to Send Data to ThingSpeak
void sendToThingSpeak(float temp, float tds, float ec, float turb, bool isDrinkable) {
  if (client.connect(server, 80)) {
    String postStr = apiKey;
    postStr += "&field1=" + String(temp, 2);
    postStr += "&field2=" + String(tds, 2);
    postStr += "&field3=" + String(ec, 4);
    postStr += "&field4=" + String(turb, 2);
    postStr += "&field5=" + String(isDrinkable ? 1 : 0);
    postStr += "\r\n\r\n";

    client.print("POST /update HTTP/1.1\n");
    client.print("Host: api.thingspeak.com\n");
    client.print("Connection: close\n");
    client.print("X-THINGSPEAKAPIKEY: " + apiKey + "\n");
    client.print("Content-Length: " + String(postStr.length()) + "\n\n");
    client.print(postStr);

    delay(500);
    Serial.println("Data sent to ThingSpeak.");
  } else {
    Serial.println("Failed to connect to ThingSpeak.");
  }
  client.stop();
}

// Function to Send Data to Google Sheets
void sendToGoogleSheets(float temp, float tds, float ec, float turb, bool isDrinkable) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(googleScriptURL);
    http.setFollowRedirects(HTTPC_STRICT_FOLLOW_REDIRECTS);
    http.addHeader("Content-Type", "application/x-www-form-urlencoded");

    String postData = "temperature=" + String(temp, 2) +
                      "&ppm=" + String(tds, 2) +
                      "&conductivity=" + String(ec, 4) +
                      "&turbidity=" + String(turb, 2) +
                      "&drinkability=" + (isDrinkable ? "1" : "0");  // Ensure correct format

    Serial.println("🔄 Sending Data: " + postData);

    int httpResponseCode = http.POST(postData);
    Serial.println("📡 Google Sheets Response Code: " + String(httpResponseCode));

    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("✅ Server Response: " + response);
    } else {
      Serial.println("❌ HTTP Error: " + String(httpResponseCode));
    }

    http.end();
  } else {
    Serial.println("❌ WiFi Not Connected!");
  }
}



