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
            "token": "advkFktt8upC1fWpjVcv115Fb1KYnd",
            "user": "gzFsyqGs2Xf6H28tkZEQCVMyHXWWdi",
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
