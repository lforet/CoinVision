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
  establishContact();  // send a byte to establish contact until receiver responds 

}

void establishContact() {
  while (Serial.available() <= 0) {
    Serial.print('A', BYTE);   // send a capital A
    delay(1000);
  }
}

// the loop() method runs over and over again,
// as long as the Arduino has power
void loop()                     
{
        char commandbuffer[100];
        int i=0;
  	// send data only when you receive data:
	if (Serial.available() > 0) {
            while( Serial.available() && i< 99) {
                commandbuffer[i++] = Serial.read();
             }
         commandbuffer[i++]='\0';
         // read the incoming byte:
	 //incomingByte = Serial.read();
	 // say what you got:
         //if(i>0){
         //   Serial.print("I received: ");
         //   Serial.println((char*)commandbuffer);
         //}
         //if(i>0){
         //    if commandbuffer == "a"
         //      Serial.print("starting motor...");  
            
          //}
		//Serial.println(incomingByte, DEC);
                //while(var < 100){
                // now do something 
                      //var++;
                      //Serial.print("Hey Andrea Count: ");
                      //Serial.println(var, DEC);
                      //delay(10); 
                      //sum = sum + var^2;
                      //Serial.print("Sum: ");
                      //Serial.println(sum, DEC);      
                //}
                //var = 0;
	}
  //rotateLeft(150, 500);
  //rotateRight(50, 1000);
  //rotateRight(150, 1000);
  //rotateRight(200, 1000);
  //rotateLeft(255, 500);
  //rotateRight(10, 1500);
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
