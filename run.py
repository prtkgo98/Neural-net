import numpy as np

def sigmd(x):
    return 1.0/(1+ np.exp(-x))

def sigmd_deri(x):
    return np.multiply( x,  (1.0 - x) )

# BEGIN


inpt=[]
y=[] 
wt1 = []
wt2 = []
layer1= []
c=[]

for i in range(0,8):
    tnp = raw_input()
    c = [ int(x) for x in tnp.split(" ") ]
    inpt.append(c)

for i in range(0,8):
    y.append(inpt[i][3])

inpt  = np.delete(inpt, -1, axis=1)



inpt = np.asmatrix(inpt)
y = np.transpose(np.asmatrix(y))


output = np.zeros(y.shape)
wt1 = np.asmatrix( np.random.rand(inpt.shape[1],6) )
wt2 = np.asmatrix( np.random.rand(6,1) )

# print(wt1)
# print(" ")
# print(wt2)
# print(" ")

for i in range(5000):

    #fORWARD
    layer1 = np.asmatrix( sigmd( np.matmul(inpt, wt1) ) )
    output = np.asmatrix( sigmd( np.matmul(layer1, wt2) ) )
    # print(layer1)
    # print(" ")
    # print(output)
    # print(" ")

    # print(y.shape)
    # print(output.shape)
    # print((y - output).shape)

    #backprop
    d_wt2 =  np.matmul (  np.transpose( np.asmatrix(layer1) )  , np.multiply( (y - output) , sigmd_deri(output) )  ) 

    c = np.multiply( (y - output),sigmd_deri(output) )
    a = np.matmul( c , np.transpose(wt2) )
    b = np.multiply ( a , sigmd_deri(layer1) )
    d_wt1 = np.matmul( np.transpose(inpt) ,  b ) 

    wt1 =  wt1 + d_wt1
    wt2 =  wt2 + d_wt2

print("TRUE OUTPUT")
print(y)
print(" ")
print("PREDICTED OUTPUT")
print(output)

