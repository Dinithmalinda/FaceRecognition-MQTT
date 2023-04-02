import time as t
import json
import AWSIoTPythonSDK.MQTTLib as AWSIoTPyMQTT

# Define ENDPOINT, CLIENT_ID, PATH_TO_CERTIFICATE, PATH_TO_PRIVATE_KEY, PATH_TO_AMAZON_ROOT_CA_1, MESSAGE, TOPIC, and RANGE
ENDPOINT = "at78pes3uyukk-ats.iot.us-west-2.amazonaws.com"
CLIENT_ID = "IPCAMdata"
PATH_TO_CERTIFICATE = "cert/RaspberrypiGateway.cert.crt"
PATH_TO_PRIVATE_KEY = "cert/RaspberrypiGateway.private.key"
PATH_TO_AMAZON_ROOT_CA_1 = "cert/root-CA.crt"
MESSAGE = "Hello World"
TOPIC = "topic_1"
RANGE = 10
      
myAWSIoTMQTTClient = AWSIoTPyMQTT.AWSIoTMQTTClient(CLIENT_ID) 
myAWSIoTMQTTClient.configureEndpoint(ENDPOINT, 8883)
myAWSIoTMQTTClient.configureCredentials(PATH_TO_AMAZON_ROOT_CA_1, PATH_TO_PRIVATE_KEY, PATH_TO_CERTIFICATE)
myAWSIoTMQTTClient.connect()

def test():
      for i in range (RANGE):
        data = "{} [{}]".format(MESSAGE, i+1)
        message = {"message" : data}
        myAWSIoTMQTTClient.publish(TOPIC, json.dumps(message), 1) 
        print("Published: '" + json.dumps(message) + "' to the topic: " + "'test/testing'")
        t.sleep(0.1)

def PUBLISH(TOPIC1,DATA1):
        myAWSIoTMQTTClient.publish(TOPIC1, json.dumps(DATA1), 1) 

def Disconnect():
        print('Publish End')
        myAWSIoTMQTTClient.disconnect()