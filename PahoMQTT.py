import paho.mqtt.client as mqtt #import the client1
import json
############
def on_message(client, userdata, message):
    print("message received " ,str(message.payload.decode("utf-8")))
    print("message topic=",message.topic)
    print("message qos=",message.qos)
    print("message retain flag=",message.retain)
########################################


client = mqtt.Client("A9GTEST") #create new instance
client.username_pw_set("A9GTEST", "A9GTEST")
client.on_message=on_message #attach function to callback
print("connecting to broker")
client.connect("122.255.9.5") #connect to broker
client.publish("IPCAMdata","dfh")
client.subscribe("SUBSCRIBETOPIC")
print("client coected")
########################################

def PUBLISH(TOPIC1,DATA1):
       print(TOPIC1,DATA1)
       client.publish(TOPIC1,json.dumps(DATA1))

def Disconnect():
        client.loop_stop() 