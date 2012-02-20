// To use, connect the Arduino to a computer and send commands using a serial terminal.
// eg AR40#   motor A forwards with a speed of 40
 
#define PwmPinMotorA 5
#define PwmPinMotorB 11
#define DirectionPinMotorA 6
#define DirectionPinMotorB 13
#define SerialSpeed 9600
#define BufferLength 16
#define LineEnd '#'
 
char inputBuffer[BufferLength];

//2-Way motor control

int incomingByte = 0;	// for incoming serial data
int var = 0;
int motorPin1 =  5;    // One motor wire connected to digital pin 5
int motorPin2 =  6;    // One motor wire connected to digital pin 6


// The setup() method runs once, when the sketch starts

void setup()   {                
  // initialize the digital pins as an output:
  pinMode(motorPin1, OUTPUT); 
  pinMode(motorPin2, OUTPUT);  
  // Setup serial com
  Serial.begin(9600);	// opens serial port, sets data rate to 9600 bps
  Serial.flush();
  establishContact();  // send a byte to establish contact until receiver responds 

}

void establishContact() {
  while (Serial.available() <= 0) {
    Serial.print('A', BYTE);   // send a capital A
    delay(1000);
  }
}


void loop()
{ 
  // get a command string form the serial port
  int inputLength = 0;
  do {
    while (!Serial.available()); // wait for input
    inputBuffer[inputLength] = Serial.read(); // read it in
  } while (inputBuffer[inputLength] != LineEnd && ++inputLength < BufferLength);
  inputBuffer[inputLength] = 0; //  add null terminator
  HandleCommand(inputBuffer, inputLength);
}


  //rotateLeft(150, 500);
  //rotateRight(50, 1000);
  //rotateRight(150, 1000);
  //rotateRight(200, 1000);
  //rotateLeft(255, 500);
  //rotateRight(10, 1500);
// process a command string
void HandleCommand(char* input, int length)
{
  Serial.println(input);
  if (length < 2) { // not a valid command
    return;
  }
  int value = 0;
  // calculate number following command
  if (length > 2) {
    value = atoi(&input[2]);
  }
  int* command = (int*)input;
  // check commands
  // note that the two bytes are swapped, ie 'RA' means command AR
  switch(*command) {
    case 'FA':
      rotateLeft(150, 500);
      delay(500);
      rotateRight(50, 1000);
       Serial.println("motor on");
      //analogWrite(PwmPinMotorA, value);
      //digitalWrite(DirectionPinMotorA, HIGH);
      break;
    case 'RA':
      // motor A reverse
      analogWrite(PwmPinMotorA, value);
      digitalWrite(DirectionPinMotorA, LOW);
      break;
    case 'FB':
      // motor B forwards
      analogWrite(PwmPinMotorB, value);
      digitalWrite(DirectionPinMotorB, LOW);
      break;
    case 'RB':
      // motor B reverse
      analogWrite(PwmPinMotorB, value);
      digitalWrite(DirectionPinMotorB, HIGH);
      break;
    default:
      break;
  }  
} 

void rotateLeft(int speedOfRotate, int length){
  analogWrite(motorPin1, speedOfRotate); //rotates motor
  digitalWrite(motorPin2, LOW);    // set the Pin motorPin2 LOW
  delay(length); //waits
  digitalWrite(motorPin1, LOW);    // set the Pin motorPin1 LOW
}

void rotateRight(int speedOfRotate, int length){
  analogWrite(motorPin2, speedOfRotate); //rotates motor
  digitalWrite(motorPin1, LOW);    // set the Pin motorPin1 LOW
  delay(length); //waits
  digitalWrite(motorPin2, LOW);    // set the Pin motorPin2 LOW
}

void rotateLeftFull(int length){
  digitalWrite(motorPin1, HIGH); //rotates motor
  digitalWrite(motorPin2, LOW);    // set the Pin motorPin2 LOW
  delay(length); //waits
  digitalWrite(motorPin1, LOW);    // set the Pin motorPin1 LOW
}

void rotateRightFull(int length){
  digitalWrite(motorPin2, HIGH); //rotates motor
  digitalWrite(motorPin1, LOW);    // set the Pin motorPin1 LOW
  delay(length); //waits
  digitalWrite(motorPin2, LOW);    // set the Pin motorPin2 LOW
}
