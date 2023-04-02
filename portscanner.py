import socket

def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            # doesn't even have to be reachable
            s.connect(('10.254.254.254', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

def pscan(ip):    
    s = socket.socket()
    try:
        con = s.connect((ip,554))
        return True
    except:
        return False

def getcamip():
    local_ip=get_ip().split(".")

    for x in range(1,30):
        y= local_ip[0]+'.'+local_ip[1]+'.'+local_ip[2]+'.'+str(x)
        print(y)
        if pscan(y):
             print(y,'Camera detected')
             return y




