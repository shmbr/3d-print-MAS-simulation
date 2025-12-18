#include <Arduino.h>

#if defined(ESP32)
  HardwareSerial printerSerial(2); // UART2
  #define RX_PIN 16
  #define TX_PIN 17
#else
  #define printerSerial Serial
#endif

PrinterState printerState = IDLE;

// Parsed data
float hotendTemp = 0;
float bedTemp = 0;
float xPos = 0, yPos = 0, zPos = 0;
int progress = 0;

String serialBuffer = "";

/* -------------------- SEND GCODE -------------------- */
void sendGcode(const char* cmd) {
  printerSerial.print(cmd);
  printerSerial.print("\n");
}

/* -------------------- PARSE RESPONSES -------------------- */
void parseLine(String line) {
  line.trim();

  // Temperature response (M105)
  // Example: ok T:200.0 /210.0 B:60.0 /60.0
  if (line.indexOf("T:") >= 0 && line.indexOf("B:") >= 0) {
    hotendTemp = line.substring(line.indexOf("T:") + 2).toFloat();
    bedTemp = line.substring(line.indexOf("B:") + 2).toFloat();
  }

  // Position response (M114)
  // Example: X:10.00 Y:20.00 Z:0.30
  if (line.indexOf("X:") >= 0) {
    xPos = line.substring(line.indexOf("X:") + 2).toFloat();
    yPos = line.substring(line.indexOf("Y:") + 2).toFloat();
    zPos = line.substring(line.indexOf("Z:") + 2).toFloat();
  }

  // Progress (M27)
  // Example: SD printing byte 12345/67890
  if (line.indexOf("SD printing byte") >= 0) {
    int slash = line.indexOf("/");
    int space = line.lastIndexOf(" ", slash);
    int current = line.substring(space + 1, slash).toInt();
    int total = line.substring(slash + 1).toInt();
    progress = (current * 100) / total;
    printerState = PRINTING;
  }

  // Printer idle
  if (line == "ok" && printerState != PRINTING) {
    printerState = IDLE;
  }
}

/* -------------------- READ SERIAL -------------------- */
void readPrinter() {
  while (printerSerial.available()) {
    char c = printerSerial.read();
    if (c == '\n') {
      parseLine(serialBuffer);
      serialBuffer = "";
    } else {
      serialBuffer += c;
    }
  }
}

/* -------------------- SETUP -------------------- */
void setup() {
  Serial.begin(115200);

#if defined(ESP32)
  printerSerial.begin(115200, SERIAL_8N1, RX_PIN, TX_PIN);
#else
  printerSerial.begin(115200);
#endif

  delay(2000);

  sendGcode("M115"); // Firmware info
}

/* -------------------- LOOP -------------------- */
unsigned long lastPoll = 0;

void loop() {
  readPrinter();

  if (millis() - lastPoll > 2000) {
    sendGcode("M105"); // temps
    sendGcode("M114"); // position
    sendGcode("M27");  // progress
    lastPoll = millis();
  }

  // Example output
  Serial.printf(
    "State:%d | Hotend:%.1f | Bed:%.1f | X:%.2f Y:%.2f Z:%.2f | Progress:%d%%\n",
    printerState, hotendTemp, bedTemp, xPos, yPos, zPos, progress
  );

  delay(10);
}
