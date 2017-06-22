from collections import defaultdict
from gtnlplib import constants

theta_hand_original = defaultdict(float,
                         {('worldnews','worldnews'):1.,
                          ('worldnews','news'):.5,
                          ('worldnews','world'):.5,
                          ('science','science'):1.,
                          ('askreddit','askreddit'):1.,
                          ('askreddit','ask'):0.5,
                          ('iama','iama'):1,
                          ('todayilearned','til'):1.,
                          ('todayilearned','todayilearned'):1.,
                          ('iama',constants.OFFSET):0.1
                         })

theta_hand = defaultdict(float,
                          {('worldnews','worldnews'):1.,
                          ('worldnews','news'):.5,
                          ('worldnews','world'):.5,
                          ('science','science'):1.,
                          ('askreddit','askreddit'):1.,
                          ('askreddit','ask'):0.75,
                          ('iama','iama'):1,
                          ('todayilearned','til'):1.,
                          ('todayilearned','todayilearned'):1.,
                          ('iama',constants.OFFSET):0.35,
                          ('science','research'):0.6,
                          ('science','ebv'):0.6,
                          ('science','pollution'):0.6,
                          ('science','psychopath'):0.6,
                          ('science','evolution'):0.6,
                          ('science','study'):0.6,
                          ('worldnews', 'ai'):0.7,
                          ('worldnews', 'ukraine'):0.6,
                          ('worldnews', 'plane'):0.6,
                          ('worldnews', 'russia'):0.6,
                          ('worldnews', 'muslim'):0.6
                         })