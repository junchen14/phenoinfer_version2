global a
a=[]

def set_value(value):
    global a
    a.extend(value)

def get_value():
    global a
    return a

if __name__ == "__main__":
    set_value([10])
    print(get_value())
    set_value([15])
    print(get_value())