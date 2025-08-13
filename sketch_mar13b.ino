const int turbidityPin = A4;  // Analog pin for turbidity sensor

void setup() {
  Serial.begin(115200);
  pinMode(turbidityPin, INPUT);
}

void loop() {
  int sum = 0;
  int numReadings = 5;

  // Take 5 readings and sum them
  for (int i = 0; i < numReadings; i++) {
    sum += analogRead(turbidityPin);
    delay(200);  // Short delay between readings
  }

  // Calculate the average sensor value
  int avgSensorValue = sum / numReadings;

  // Map the average value from 220-250 to 0-8
  int turbidity = map(avgSensorValue, 220, 280, 0, 8);

  // Ensure the value stays within the 0-8 range
  turbidity = constrain(turbidity, 0, 8);

  // Print the result
 
  Serial.println(turbidity);

  delay(100);  // Wait 1 second before next reading
}
