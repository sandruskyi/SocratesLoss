"""
@User: sandruskyi
"""
hidden_features = None

def set_hidden_features(value):
    global hidden_features
    hidden_features = value

def get_hidden_features():
    global hidden_features
    return hidden_features