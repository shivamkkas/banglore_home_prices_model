import json
import pickle
import os
import numpy as np
__locations = None
__data_columns = None
__model = None

def get_estimated_price(location,sqft,bhk,bath):
    global x
    if(__data_columns and __model) is None:
        load_saved_artifacts()

        try:
            loc_index = __data_columns.index(location.lower())
        except:
            loc_index = -1    
        x = np.zeros(len(__data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk
        if loc_index >= 0:
            x[loc_index] = 1

           # Use your model to predict the price
    return round(__model.predict([x])[0],2)


def get_location_name():
    global __locations
    if __locations is None:
         load_saved_artifacts() # Doing this will make first request dependent on the execution time of this function call 
    return __locations



def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __data_columns
    global __locations

    with open("./artificate/columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    
    global __model
    with open("./artificate/banglore_home_prices_model.pickle",'rb') as f:
        __model = pickle.load(f)  



    print("loading saved artifacts...done") 





if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_name())

    print(get_estimated_price('Rajaji Nagar',1179,2,2))
    print(get_estimated_price('1st Phase JP Nagar',1000,2,2))
    print(get_estimated_price('1st Phase JP Nagar',1000,3,3))
