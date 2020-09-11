from som import SOM

if __name__ == "__main__":
    a=SOM(5,5,3,2,False,0.05)
    a.train(100,[[[0,0,0],[0,0]],[[1,0,0],[0,1]],[[0,1,0],[1,0]],[[1,1,1],[1,1]]])
    print ("Prediction 0 and 0 ,", round(a.predict([0,0,0])[0,1]))