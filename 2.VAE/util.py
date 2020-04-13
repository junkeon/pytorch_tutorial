from slacker import Slacker
import hyperparams as hp

token = hp.token
channel = hp.channel

slack = Slacker(token) if token else None

def logging(msg, opt=0):
    if opt:
        print(msg)
    if slack is not None:
        slack.chat.post_message(channel, msg)