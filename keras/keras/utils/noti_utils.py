import httplib, urllib
import time

def notify(*msg):
    try:
        if isinstance(msg, tuple):
            msg = " ".join([str(m) for m in msg])
        print msg
        conn = httplib.HTTPSConnection("api.pushover.net:443")
        conn.request("POST", "/1/messages.json",
          urllib.urlencode({
            "token": "a13cfpBJn78CCN6Qz5UhDzCEAFcSE9",
            "user": "uHnR1kmckN2CKQdFi5NBVYDcsFC1A6",
            "message": msg,
            "timestamp": int(time.time())
          }), { "Content-type": "application/x-www-form-urlencoded" })
        conn.getresponse()
    except Exception as e:
        print  e, "but Pass! in notify()"
        pass

def main():
    notify('test!')

if __name__ == '__main__':
    main()
